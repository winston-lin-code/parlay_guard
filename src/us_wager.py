"""Match a scanner selection to a Polymarket US contract and price a limit order."""

import re
from decimal import Decimal, InvalidOperation, ROUND_CEILING
from secrets import token_hex
from urllib.parse import urlencode

import requests

from .inplay_scan import PROBABILITY_HAIRCUT, utc

GATEWAY = "https://gateway.polymarket.us"
SLUG = re.compile(r"[a-z0-9][a-z0-9-]{0,159}")
ONE = Decimal(1)
CENT = Decimal(".01")
STANDARD_FEE = Decimal(".0695")
BOOK_MAX_AGE = 15
EVENT_CACHE_MAX_AGE = 30


class WagerError(ValueError):
    pass


def decimal(value):
    try:
        number = Decimal(str(value))
        if number.is_finite():
            return number
    except InvalidOperation:
        pass
    raise WagerError("The exchange did not provide a valid price or quantity.")


def public(path, *, fresh_book=False, fresh_score=False):
    # Event metadata has a normal 30-second CDN lifetime. The book must be newer:
    # the gateway can serve a cached book even with a no-cache request header.
    options = {"params": {"_": token_hex(8)}} if fresh_book or fresh_score else {}
    response = requests.get(GATEWAY + path, timeout=(4, 6), headers={"Cache-Control": "no-cache"}, **options)
    if response.status_code == 404:
        raise WagerError("The matching Polymarket US listing is unavailable.")
    response.raise_for_status()
    max_age = BOOK_MAX_AGE if fresh_book or fresh_score else EVENT_CACHE_MAX_AGE
    if float(response.headers.get("Age", 0)) > max_age:
        if fresh_book:
            raise WagerError("Polymarket US returned a stale order book. Get a new US quote.")
        if fresh_score:
            raise WagerError("Polymarket US returned cached game state.")
        raise WagerError("The Polymarket US listing is out of date. Get a new US quote.")
    return response.json()


def name(value):
    return re.sub(r"[^a-z0-9]", "", str(value or "").lower())


def team_names(side):
    team = side.get("team") or {}
    return {name(team.get(key)) for key in
            ("name", "safeName", "alias", "abbreviation", "displayAbbreviation")} - {""}


def match_contract(pick, event, start):
    """Match outcome, full-game scope, signed line, fixture slug and start."""
    if event.get("slug") != pick["event"] or event.get("active") is not True or event.get("closed") is not False or not start:
        raise WagerError("No open Polymarket US event matches this game.")
    kind, side = pick["market_type"], name(pick["side"])
    target = None
    types = {"moneyline": "winner", "spreads": "spread", "totals": "total"}
    if kind not in types:
        raise WagerError("This market type has no verified Polymarket US mapping.")
    if kind != "moneyline":
        target = decimal(pick.get("line"))
        if abs(target) % 1 != Decimal(".5"):
            raise WagerError("This line has no verified Polymarket US mapping.")
    spread_team = None
    if kind == "spreads":
        match = re.fullmatch(r"Spread:\s*(.+?)\s*\(([+-]?\d+(?:\.\d+)?)\)", pick["market"], re.I)
        if not match or decimal(match[2]) != target:
            raise WagerError("The original spread rules could not be verified.")
        spread_team = name(match[1])
    matched = []
    for contract in event.get("markets", []):
        slug = contract.get("slug") or ""
        listed = utc(contract.get("gameStartTime"))
        scope = str(contract.get("sportsMarketType") or "")
        if (not SLUG.fullmatch(slug) or listed is None or abs((listed - start).total_seconds()) > 60
                or contract.get("active") is not True or contract.get("closed") is not False
                or not scope.endswith("_team_full_game_" + types[kind]) or contract.get("marketType") != kind):
            continue
        rules = str(contract.get("description") or "")
        if re.search(r"regulation only|exclud\w+ overtime", rules, re.I):
            continue
        if pick.get("half_payout_tie") and not re.search(r"tie.{0,70}(?:50-50|\$0\.50)", rules, re.I):
            continue
        sides = contract.get("marketSides") or []
        if len(sides) != 2 or {item.get("long") for item in sides} != {True, False}:
            continue
        for chosen in sides:
            if kind == "totals":
                matches = side in ("over", "under") and name(chosen.get("description")) == side and decimal(contract.get("line")) == target
            else:
                matches = side in team_names(chosen)
                if matches and kind == "spreads":
                    selected_line = target if spread_team in team_names(chosen) else -target
                    other = next(item for item in sides if item is not chosen)
                    matches = (spread_team in (team_names(chosen) | team_names(other))
                               and decimal(chosen.get("description")) == selected_line)
            if matches and chosen.get("tradable") is True:
                matched.append((contract, chosen["long"]))
    if len(matched) != 1:
        raise WagerError("No unique US contract matches this side and line.")
    return matched[0]


def contract_url(event, contract, long):
    """Use the exchange's market and outcome selection parameters."""
    league = re.sub(r"-\d{4}$", "", str(event.get("seriesSlug") or ""))
    event_slug = str(event.get("slug") or "")
    market_slug = str(contract.get("slug") or "")
    sides = [side for side in contract.get("marketSides", []) if side.get("long") is long]
    outcome = str(sides[0].get("id") or "") if len(sides) == 1 else ""
    if not all(SLUG.fullmatch(value) for value in (league, event_slug, market_slug)) or not outcome.isascii() or not outcome.isdigit():
        raise WagerError("Polymarket US did not provide an exact link for this selection.")
    return (f"https://polymarket.us/sports/{league}/{event_slug}?"
            + urlencode({"marketSlug": market_slug, "outcomeId": outcome}))


def quote_order(pick, contract, long, quantity, book, now, *, slippage=Decimal("0")):
    slug = contract["slug"]
    timestamp = utc(book.get("transactTime"))
    if book.get("marketSlug") != slug:
        raise WagerError("Polymarket US did not return the book for the selected contract.")
    if book.get("state") != "MARKET_STATE_OPEN":
        raise WagerError("This Polymarket US contract is closed or paused.")
    if timestamp is None or not -5 <= (now - timestamp).total_seconds() <= BOOK_MAX_AGE:
        raise WagerError("Polymarket US has no fresh price for this contract. Get a new US quote.")
    if quantity < decimal(contract.get("minimumTradeQty")):
        raise WagerError("This quantity is below the exchange minimum.")
    levels = []
    for level in book.get("offers" if long else "bids", []):
        px = level["px"]
        if px.get("currency") != "USD":
            continue
        quoted = decimal(px.get("value"))
        available = decimal(level.get("qty"))
        if CENT <= quoted <= ONE - CENT and available > 0:
            levels.append((quoted if long else ONE - quoted, available))
    remaining = quantity
    for limit, available in sorted(levels):
        remaining -= available
        if remaining <= 0:
            break
    else:
        raise WagerError("Insufficient order-book depth for this quantity.")
    tick = decimal(contract.get("orderPriceMinTickSize"))
    if tick <= 0 or limit % tick:
        raise WagerError("The quoted price does not match the US market tick size.")
    # Bound fees even if fills occur at better prices, where the fee may be higher.
    coefficient = max(STANDARD_FEE, decimal(contract.get("feeCoefficient") or STANDARD_FEE))
    effective = limit + slippage
    if not limit <= effective <= ONE:
        raise WagerError("The simulated slippage makes this contract unaffordable.")
    peak = min(effective, Decimal(".5"))
    fee_cap = (coefficient * quantity * peak * (ONE - peak)).quantize(CENT, rounding=ROUND_CEILING)
    total = effective * quantity + fee_cap
    screened = max(Decimal(0), decimal(pick["probability"]) - decimal(PROBABILITY_HAIRCUT))
    payload = {"marketSlug": slug, "type": "ORDER_TYPE_LIMIT",
               "price": {"value": str(limit if long else ONE - limit), "currency": "USD"},
               "quantity": quantity, "tif": "TIME_IN_FORCE_FILL_OR_KILL",
               "intent": "ORDER_INTENT_BUY_LONG" if long else "ORDER_INTENT_BUY_SHORT",
               "manualOrderIndicator": "MANUAL_ORDER_INDICATOR_MANUAL", "synchronousExecution": True}
    return payload, {"market": contract["question"], "side": pick["side"], "limit": str(limit),
                     "quantity": quantity, "fee_estimate": str(fee_cap), "total_estimate": str(total),
                     "screened_ev": float(screened * quantity / total - ONE),
                     "rules": contract.get("description") or "", "quote_utc": timestamp.isoformat()}
