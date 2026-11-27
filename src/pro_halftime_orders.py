"""Submit and record one automatic Polymarket US order per NHL or MLB game."""

from contextlib import closing
from decimal import Decimal
import json
from pathlib import Path
import sqlite3

from .inplay_scan import utc
from .pro_live_ev import MAX_BOOK_AGE, now

LIVE_STAKE = Decimal("5.00")
DEFAULT_LIVE_DB = Path("data/live_orders/pro_halftime.sqlite3")


class HalftimeOrders:
    def __init__(self, path, exchange_client):
        self.path = path
        self.exchange_client = exchange_client
        path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self.connect()) as store, store:
            store.execute("""CREATE TABLE IF NOT EXISTS pro_halftime_orders (
                league TEXT NOT NULL, official_game_id TEXT NOT NULL, event TEXT NOT NULL UNIQUE,
                recorded_at TEXT NOT NULL, state TEXT NOT NULL, detail TEXT NOT NULL,
                PRIMARY KEY (league, official_game_id)
            )""")

    def connect(self):
        return sqlite3.connect(self.path, timeout=5)

    def recent(self, limit=20):
        with closing(self.connect()) as store:
            return [{**json.loads(detail), "state": state} for state, detail in store.execute(
                "SELECT state, detail FROM pro_halftime_orders ORDER BY recorded_at DESC, event LIMIT ?", (limit,)
            )]

    def place(self, entry, payload):
        quote = entry["quote"]
        quoted = utc(quote["quote_utc"])
        if quoted is None or not 0 <= (now() - quoted).total_seconds() <= MAX_BOOK_AGE:
            return {"status": "quote_expired_before_order"}
        if Decimal(quote["total_estimate"]) > LIVE_STAKE:
            return {"status": "outside_live_stake_cap"}

        body = {**payload, "manualOrderIndicator": "MANUAL_ORDER_INDICATOR_AUTOMATIC"}
        detail = {**entry, "market_slug": body["marketSlug"]}
        key = entry["league"], str(entry["official_game_id"])
        with closing(self.connect()) as store:
            # Acquire the exchange client before reserving a slot. A client setup failure
            # cannot result in an order and should leave the game eligible for a later scan.
            with self.exchange_client() as exchange:
                with store:
                    cursor = store.execute("""INSERT INTO pro_halftime_orders VALUES (?, ?, ?, ?, 'pending', ?)
                        ON CONFLICT DO NOTHING""", (*key, entry["event"], entry["recorded_at"], json.dumps(detail)))
                    if cursor.rowcount == 0:
                        return {"status": "live_order_already_attempted"}

                # The reservation stays committed if the request or process fails after this point.
                if not 0 <= (now() - quoted).total_seconds() <= MAX_BOOK_AGE:
                    with store:
                        store.execute("DELETE FROM pro_halftime_orders WHERE league=? AND official_game_id=?", key)
                    return {"status": "quote_expired_before_order"}

                state = "unknown"
                try:
                    result = exchange.orders.create(body)
                    order_id = result.get("id")
                    if isinstance(order_id, str) and order_id:
                        detail["order_id"] = order_id
                        state = "submitted"
                        executions = result.get("executions") or []
                        if executions:
                            reported = (executions[-1].get("order") or {}).get("state")
                            if isinstance(reported, str) and reported:
                                state = reported
                except Exception:
                    # A missing response may still mean the exchange received the order.
                    pass
                with store:
                    store.execute("UPDATE pro_halftime_orders SET state=?, detail=? WHERE league=? AND official_game_id=?",
                                  (state, json.dumps(detail), *key))
        return {"status": "live_order_unknown" if state == "unknown" else "live_order_recorded", "state": state}
