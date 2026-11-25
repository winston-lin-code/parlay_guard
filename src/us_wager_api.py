"""User-confirmed Polymarket US order tickets for positive-EV scanner selections."""

import hmac
import json
import os
import secrets
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

from .inplay_api import scanner
from .inplay_scan import utc
from .us_wager import WagerError, decimal, match_contract, public, quote_order

router = APIRouter(prefix="/api/inplay/us-wager")
PREVIEW_SECONDS = 15


class Selection(BaseModel):
    event: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{0,159}$")
    token_id: str = Field(pattern=r"^\d{1,90}$")
    quantity: int = Field(strict=True, ge=1, le=10000)


class Confirmation(BaseModel):
    ticket: str = Field(pattern=r"^[a-f0-9]{48}$")


def configuration():
    return {name: os.getenv(key, "") for name, key in {
        "secret": "PARLAY_GUARD_ORDER_TOKEN", "origin": "PARLAY_GUARD_ORDER_ORIGIN",
        "db": "PARLAY_GUARD_ORDER_DB", "key": "POLYMARKET_KEY_ID", "api_secret": "POLYMARKET_SECRET_KEY",
    }.items()}


def enabled(config):
    return (os.getenv("PARLAY_GUARD_US_ORDERS") == "1" and len(config["secret"]) >= 32
            and all(config.values()) and Path(config["db"]).is_absolute())


def authorize(config, token, origin):
    if not enabled(config):
        raise HTTPException(403, "Direct US orders are disabled on this instance.")
    if origin != config["origin"] or not hmac.compare_digest(token.encode(), config["secret"].encode()):
        raise HTTPException(403, "Trading access denied.")


def client(config):
    from polymarket_us import PolymarketUS

    return PolymarketUS(key_id=config["key"], secret_key=config["api_secret"], timeout=6, max_retries=0)


def active_pick(selection):
    scan = scanner.read()
    matches = [row for row in scan["opportunities"]
               if row["event"] == selection.event and row["token_id"] == selection.token_id
               and row["screened_ev"] > 0 and row.get("market_type")]
    if len(matches) != 1:
        raise WagerError("This positive EV selection has expired. Return to the scanner and select a fresh quote.")
    pick = matches[0]
    starts = {utc(row.get("start_utc")) for row in scan["games"]
              if row["event"] == selection.event and row["phase"] == pick["phase"]}
    if len(starts) != 1 or None in starts:
        raise WagerError("The game start time could not be verified.")
    return pick, starts.pop()


def current_order(selection):
    pick, start = active_pick(selection)
    event = public("/v1/events/slug/" + selection.event)["event"]
    contract, long = match_contract(pick, event, start)
    book = public("/v1/markets/" + contract["slug"] + "/book")["marketData"]
    current, current_start = active_pick(selection)
    keys = ("phase", "score", "market", "market_type", "line", "side", "half_payout_tie")
    if start != current_start or any(current.get(key) != pick.get(key) for key in keys):
        raise WagerError("The game or selection changed. Request a new quote.")
    return quote_order(current, contract, long, selection.quantity, book, datetime.now(timezone.utc))


def database(config):
    store = sqlite3.connect(config["db"], timeout=3)
    store.execute("""CREATE TABLE IF NOT EXISTS wager_tickets (
        id TEXT PRIMARY KEY, expires REAL NOT NULL, selection TEXT NOT NULL,
        order_payload TEXT NOT NULL, state TEXT NOT NULL
    )""")
    return store


def checked_quote(selection):
    try:
        return current_order(selection)
    except WagerError as exc:
        raise HTTPException(409, str(exc)) from None
    except Exception:
        raise HTTPException(502, "Polymarket US pricing is temporarily unavailable.") from None


@router.get("/status")
def status():
    return {"direct_orders": bool(enabled(configuration()))}


@router.post("/preview")
def preview(selection: Selection, x_order_token: str = Header(default=""), origin: str = Header(default="")):
    config = configuration()
    direct = enabled(config)
    if direct:
        authorize(config, x_order_token, origin)
    body, summary = checked_quote(selection)
    summary["ticket"] = None
    if not direct or summary["screened_ev"] <= 0:
        return summary
    try:
        with client(config) as exchange:
            response = exchange.orders.preview({"request": body})["order"]
        if (response.get("marketSlug") != body["marketSlug"] or response.get("intent") != body["intent"]
                or response.get("quantity") != body["quantity"] or response.get("tif") != body["tif"]
                or decimal(response.get("price", {}).get("value")) != decimal(body["price"]["value"])
                or response.get("state") in ("ORDER_STATE_REJECTED", "ORDER_STATE_CANCELED")):
            raise WagerError("Exchange preview did not confirm the same order.")
        expires = utc(summary["quote_utc"]).timestamp() + PREVIEW_SECONDS
        now = datetime.now(timezone.utc).timestamp()
        if now >= expires:
            raise WagerError("The quote expired while previewing. Request a fresh quote.")
        ticket = secrets.token_hex(24)
        with closing(database(config)) as store, store:
            store.execute("DELETE FROM wager_tickets WHERE state = 'preview' AND expires < ?", (now,))
            store.execute("INSERT INTO wager_tickets VALUES (?, ?, ?, ?, 'preview')",
                          (ticket, expires, selection.model_dump_json(), json.dumps(body)))
    except WagerError as exc:
        raise HTTPException(409, str(exc)) from None
    except Exception:
        raise HTTPException(502, "The exchange could not verify the order preview.") from None
    return {**summary, "ticket": ticket, "expires": expires}


@router.post("/confirm")
def confirm(confirmation: Confirmation, x_order_token: str = Header(default=""), origin: str = Header(default="")):
    config = configuration()
    authorize(config, x_order_token, origin)
    ticket = confirmation.ticket
    try:
        with closing(database(config)) as store, store:
            now = datetime.now(timezone.utc).timestamp()
            claimed = store.execute("""UPDATE wager_tickets SET state = 'claimed'
                WHERE id = ? AND expires > ? AND state = 'preview' RETURNING selection, order_payload, expires""",
                                    (ticket, now)).fetchone()
    except sqlite3.Error:
        raise HTTPException(503, "Order storage unavailable. No order was submitted.") from None
    if not claimed:
        raise HTTPException(409, "This ticket has expired or already been confirmed.")
    selection = Selection.model_validate_json(claimed[0])
    previous = json.loads(claimed[1])
    body, summary = checked_quote(selection)
    if body != previous or summary["screened_ev"] <= 0 or datetime.now(timezone.utc).timestamp() >= claimed[2]:
        raise HTTPException(409, "Price or EV changed. Review a new quote before submitting.")
    try:
        with client(config) as exchange:
            result = exchange.orders.create(body)
        order_id = result.get("id")
        if not isinstance(order_id, str) or not order_id:
            raise ValueError("Missing exchange order ID")
        executions = result.get("executions") or []
        order = (executions[-1].get("order") or {}) if executions else {}
        order_state = order.get("state") or "submitted"
        return {"id": order_id, "state": order_state}
    except Exception:
        raise HTTPException(502, "Order status is unknown. Check Polymarket US activity before placing another order.") from None
