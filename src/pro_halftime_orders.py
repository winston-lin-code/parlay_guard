"""Reserve and record automatic Polymarket US orders by game."""

from contextlib import closing
from decimal import Decimal
import json
from pathlib import Path
import sqlite3

from .inplay_scan import utc
from .pro_live_ev import MAX_BOOK_AGE, now

LIVE_STAKE = Decimal("5.00")
DEFAULT_LIVE_DB = Path("data/live_orders/pro_halftime.sqlite3")
EXPIRED = "ORDER_STATE_EXPIRED"
UNCONFIRMED_STATES = ("submitted", "ORDER_STATE_NEW", "ORDER_STATE_PENDING_NEW",
                      "ORDER_STATE_PARTIALLY_FILLED", "ORDER_STATE_PENDING_CANCEL")


def update_from_order(detail, order):
    if not isinstance(order, dict):
        return None
    state, filled = order.get("state"), order.get("cumQuantity")
    if not isinstance(state, str) or not state.startswith("ORDER_STATE_"):
        return None
    requested = detail.get("quantity", detail["quote"].get("quantity"))
    if type(filled) is int and type(requested) is int and 0 <= filled <= requested:
        detail["filled_quantity"] = filled
    return state


def confirmed_zero_fill_expiry(exchange, attempt):
    try:
        order = exchange.orders.retrieve(attempt["order_id"]).get("order")
    except Exception:
        return False
    return (isinstance(order, dict) and order.get("id") == attempt["order_id"]
            and order.get("marketSlug") == attempt["market_slug"] and order.get("state") == EXPIRED
            and type(order.get("cumQuantity")) is int and order["cumQuantity"] == 0)


class AutomaticOrders:
    def __init__(self, path, exchange_client, *, max_book_age=MAX_BOOK_AGE, retry_expired=False):
        self.path = path
        self.exchange_client = exchange_client
        self.max_book_age = max_book_age
        self.retry_expired = retry_expired
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
            rows = store.execute("SELECT state, detail FROM pro_halftime_orders ORDER BY recorded_at DESC, event LIMIT ?",
                                 (limit,)).fetchall()
        attempts = []
        for state, raw in rows:
            detail = json.loads(raw)
            if previous := detail.pop("previous_attempt", None):
                attempts.append(previous)
                detail["attempt_number"] = 2
            attempts.append({**detail, "state": state})
        attempts.sort(key=lambda row: row["recorded_at"], reverse=True)
        return attempts if limit < 0 else attempts[:limit]

    def attempts(self, league, game_ids):
        if not game_ids:
            return {}
        placeholders = ",".join("?" for _ in game_ids)
        with closing(self.connect()) as store:
            return {game_id: {**json.loads(detail), "state": state}
                    for game_id, state, detail in store.execute(
                        f"SELECT official_game_id, state, detail FROM pro_halftime_orders "
                        f"WHERE league=? AND official_game_id IN ({placeholders})", (league, *game_ids))}

    def can_retry(self, attempt):
        return (self.retry_expired and attempt["state"] == EXPIRED and bool(attempt.get("order_id"))
                and attempt.get("filled_quantity") in (None, 0) and not attempt.get("previous_attempt"))

    def refresh(self):
        placeholders = ",".join("?" for _ in UNCONFIRMED_STATES)
        with closing(self.connect()) as store:
            pending = store.execute(f"""SELECT league, official_game_id, state, detail FROM pro_halftime_orders
                WHERE state IN ({placeholders}) ORDER BY recorded_at DESC LIMIT 5""", UNCONFIRMED_STATES).fetchall()
        if not pending:
            return
        with self.exchange_client() as exchange:
            for league, game_id, previous, raw in pending:
                detail = json.loads(raw)
                order = exchange.orders.retrieve(detail["order_id"]).get("order")
                if (not isinstance(order, dict) or order.get("id") != detail["order_id"]
                        or order.get("marketSlug") != detail["market_slug"]):
                    continue
                if state := update_from_order(detail, order):
                    with closing(self.connect()) as store, store:
                        store.execute("""UPDATE pro_halftime_orders SET state=?, detail=?
                            WHERE league=? AND official_game_id=? AND state=? AND detail=?""",
                                      (state, json.dumps(detail), league, game_id, previous, raw))

    def place(self, entry, payload, *, retry_for=None):
        quote = entry["quote"]
        quoted = utc(quote["quote_utc"])
        if quoted is None or not 0 <= (now() - quoted).total_seconds() <= self.max_book_age:
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
                previous = None
                with store:
                    if retry_for is None:
                        cursor = store.execute("""INSERT INTO pro_halftime_orders VALUES (?, ?, ?, ?, 'pending', ?)
                            ON CONFLICT DO NOTHING""", (*key, entry["event"], entry["recorded_at"], json.dumps(detail)))
                    else:
                        previous = store.execute("""SELECT recorded_at, state, detail FROM pro_halftime_orders
                            WHERE league=? AND official_game_id=? AND event=?""", (*key, entry["event"])).fetchone()
                        if previous is None:
                            return {"status": "live_order_already_attempted"}
                        previous_at, previous_state, previous_raw = previous
                        old = {**json.loads(previous_raw), "state": previous_state}
                        if (not self.can_retry(old) or old["order_id"] != retry_for
                                or old["market_slug"] != body["marketSlug"]):
                            return {"status": "live_order_already_attempted"}
                        cutoff = utc(previous_at)
                        if cutoff is None or quoted <= cutoff:
                            return {"status": "quote_expired_before_order"}
                        if not confirmed_zero_fill_expiry(exchange, old):
                            return {"status": "live_order_retry_unconfirmed"}
                        detail["previous_attempt"] = {**old, "filled_quantity": 0}
                        cursor = store.execute("""UPDATE pro_halftime_orders SET recorded_at=?, state='pending', detail=?
                            WHERE league=? AND official_game_id=? AND state=? AND detail=?""",
                                               (entry["recorded_at"], json.dumps(detail), *key, previous_state, previous_raw))
                    if cursor.rowcount == 0:
                        return {"status": "live_order_already_attempted"}

                # The reservation stays committed if the request or process fails after this point.
                if not 0 <= (now() - quoted).total_seconds() <= self.max_book_age:
                    with store:
                        if previous is None:
                            store.execute("DELETE FROM pro_halftime_orders WHERE league=? AND official_game_id=?", key)
                        else:
                            store.execute("""UPDATE pro_halftime_orders SET recorded_at=?, state=?, detail=?
                                WHERE league=? AND official_game_id=?""", (*previous, *key))
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
                            order = executions[-1].get("order") or {}
                            if (order.get("id", order_id) == order_id
                                    and order.get("marketSlug", detail["market_slug"]) == detail["market_slug"]):
                                state = update_from_order(detail, order) or state
                except Exception:
                    # A missing response may still mean the exchange received the order.
                    pass
                with store:
                    store.execute("UPDATE pro_halftime_orders SET state=?, detail=? WHERE league=? AND official_game_id=?",
                                  (state, json.dumps(detail), *key))
        return {"status": "live_order_unknown" if state == "unknown" else "live_order_recorded", "state": state}


HalftimeOrders = AutomaticOrders
