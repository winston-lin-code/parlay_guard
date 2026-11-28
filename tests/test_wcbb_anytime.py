from base64 import b64encode
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import sys
from threading import Barrier, Event
from types import SimpleNamespace

import pytest

from src import env_config, pro_halftime_orders, run_research, wcbb_anytime as paper
from src.inplay_api import ScanState
from src.tennis_wager import BookStream
from src.wcbb_all_features import FEATURES, MODEL_ID, SEASON, AllFeatures

NOW = datetime(2026, 11, 27, 18, tzinfo=timezone.utc)
EVENT = "wcbb-duke-unc-2026-11-27"
MARKET = "aec-" + EVENT


def fixture(period=1, score=16):
    return {
        "id": "40123", "date": (NOW - timedelta(minutes=20)).isoformat(),
        "competitions": [{
            "status": {"period": period, "clock": 100, "displayClock": "1:40",
                       "type": {"state": "in", "name": "STATUS_IN_PROGRESS"}},
            "competitors": [
                {"homeAway": "home", "score": str(score),
                 "team": {"id": "1", "abbreviation": "UNC", "shortDisplayName": "North Carolina"}},
                {"homeAway": "away", "score": "10",
                 "team": {"id": "2", "abbreviation": "DUKE", "shortDisplayName": "Duke"}},
            ],
        }],
    }


def inputs(period, probability, bid, size=20):
    contract = {
        "slug": MARKET, "question": "Duke vs North Carolina",
        "active": True, "closed": False, "marketType": "moneyline",
        "sportsMarketType": "basketball_team_full_game_winner", "description": "Overtime is included.",
        "gameStartTime": fixture()["date"], "orderPriceMinTickSize": ".01", "minimumTradeQty": "1",
        "feeCoefficient": ".11",
        "marketSides": [
            {"long": False, "tradable": True, "team": {"name": "North Carolina (W)"}},
            {"long": True, "tradable": True, "team": {"name": "Duke (W)"}},
        ],
    }
    summary = {
        "header": {"id": "40123"},
        "plays": [{"id": "play-1", "wallclock": (NOW - timedelta(seconds=20)).isoformat(),
                   "period": {"number": period}, "homeScore": 16, "awayScore": 10}],
        "winprobability": [{"playId": "play-1", "homeWinPercentage": probability}],
    }
    book = {"marketSlug": MARKET, "state": "MARKET_STATE_OPEN", "transactTime": NOW.isoformat(),
            "bids": [{"px": {"currency": "USD", "value": bid}, "qty": str(size)}]}
    return contract, summary, book


@pytest.mark.parametrize(("period", "probability", "bid", "status"), [
    (1, .73, ".50", "pending_recheck"),
    (4, .73, ".50", "pending_recheck"),
    (5, .73, ".50", "pending_recheck"),
    (3, .60, ".50", "below_probability"),
    (3, .73, ".81", "below_price"),
    (3, .66, ".40", "below_ev"),
])
def test_anytime_entry_thresholds_and_depth(monkeypatch, period, probability, bid, status):
    contract, summary, book = inputs(period, probability, bid, size=4)
    monkeypatch.setattr(paper, "now", lambda: NOW)
    monkeypatch.setattr(paper, "read_json", lambda *args, **kwargs: summary)
    monkeypatch.setattr(paper, "public", lambda *args, **kwargs: {"marketData": book})
    row, entry = paper.scan_game(paper.game_state(fixture(period)), [(EVENT, contract)], {})
    assert row["status"] == status
    if status == "pending_recheck":
        assert entry["quantity"] == 4
        assert entry["cost"] == "2.15"
        assert entry["side"] == "North Carolina"
        assert entry["long"] is False
        assert entry["screened_ev"] >= .05
    else:
        assert entry is None


def test_rejects_newer_or_changed_state(monkeypatch):
    _, summary, _ = inputs(4, .75, ".50")
    game = paper.game_state(fixture(4))
    assert paper.forecast(summary, game, NOW)
    summary["plays"].append({**summary["plays"][0], "id": "play-2", "homeScore": 18,
                             "wallclock": (NOW - timedelta(seconds=2)).isoformat()})
    assert paper.forecast(summary, game, NOW) is None
    summary["plays"] = [summary["plays"][-1]]
    assert paper.forecast(summary, paper.game_state(fixture(4, 18)), NOW) is None


def test_all_features_qualifies_first_confirmed_grid_and_uses_no_espn_estimate(monkeypatch):
    contract, summary, book = inputs(1, .01, ".55")
    game = paper.game_state(fixture())
    summary["plays"] = [
        {"wallclock": (NOW - timedelta(seconds=seconds)).isoformat(), "period": {"number": 1},
         "clock": {"displayValue": clock}, "homeScore": 16, "awayScore": 10}
        for seconds, clock in ((45, "2:10"), (20, "1:59"), (10, "1:40"))
    ]
    n = len(FEATURES)
    artifact = {"model": MODEL_ID, "version": 1, "season": SEASON, "features": list(FEATURES),
                "median": [0] * n, "mean": [0] * n, "scale": [1] * n, "weights": [0] * n,
                "intercept": .4054651081081642, "generated_utc": (NOW - timedelta(days=1)).isoformat(),
                "context": {"1": [[(NOW - timedelta(days=2)).timestamp(), 1520, 101, 96, 1505]]},
                "members": [1, 2], "historical_elo": {"2": 1500}, "home_venues": {}}
    model = AllFeatures(artifact)
    assert model.forecast(summary, game, NOW, delay=15)[0] == Decimal(".60")
    monkeypatch.setattr(paper, "now", lambda: NOW)
    monkeypatch.setattr(paper, "read_json", lambda *args, **kwargs: summary)
    monkeypatch.setattr(paper.SCANNER_BOOKS, "read", lambda *args, **kwargs: book)
    row, entry = paper.scan_game(game, [(EVENT, contract)], {}, model=model)
    assert row["status"] == "pending_recheck"
    assert entry["side"] == "North Carolina" and entry["strategy"] == MODEL_ID
    summary["plays"][-1]["homeScore"] = 18
    row, entry = paper.scan_game(game, [(EVENT, contract)], {}, model=model)
    assert row["status"] == "waiting_for_probability" and entry is None


@pytest.mark.parametrize(("failure", "expected"), [
    ("espn", "espn_probability_unavailable"),
    ("book", "book_unavailable"),
])
def test_verified_market_link_survives_feed_failure(monkeypatch, failure, expected):
    contract, summary, _ = inputs(1, .73, ".50")

    def unavailable(*args, **kwargs):
        raise ValueError("feed unavailable")

    monkeypatch.setattr(paper, "read_json", unavailable if failure == "espn" else lambda *args, **kwargs: summary)
    monkeypatch.setattr(paper, "public", unavailable)
    game = paper.game_state(fixture())
    row, entry = paper.scan_game(game, [(EVENT, contract)], {})
    assert row["status"] == expected and entry is None
    assert row["market_url"] == f"https://polymarket.us/sports/wcbb/{EVENT}?marketSlug={MARKET}"
    assert "market_url" not in paper.scan_game(game, [], {})[0]
    assert paper.scan_game(game, None, {})[0]["status"] == "market_catalog_unavailable"


def test_wcbb_distinguishes_rate_limits_and_stale_books(monkeypatch):
    contract, summary, book = inputs(1, .73, ".50")
    monkeypatch.setattr(paper, "now", lambda: NOW)
    monkeypatch.setattr(paper, "read_json", lambda *args, **kwargs: summary)
    game = paper.game_state(fixture())

    def limited(*args, **kwargs):
        raise paper.BookRateLimited(10)

    monkeypatch.setattr(paper.SCANNER_BOOKS, "read", limited)
    row, entry = paper.scan_game(game, [(EVENT, contract)], {})
    assert (row["status"], row["retry_after_seconds"], entry) == ("book_rate_limited", 10, None)
    book["transactTime"] = (NOW - timedelta(seconds=64)).isoformat()
    monkeypatch.setattr(paper.SCANNER_BOOKS, "read", lambda *args, **kwargs: book)
    row, entry = paper.scan_game(game, [(EVENT, contract)], {})
    assert (row["status"], row["book_age_seconds"], entry) == ("book_stale", 64, None)


def test_wcbb_prices_verified_stream_book_without_rest(monkeypatch):
    contract, summary, book = inputs(1, .73, ".50")
    stream = BookStream(None, Event(), request_id="wcbb-anytime")
    stream.set_markets([MARKET])
    stream.publish(stream.markets, {**book, "offers": []})
    monkeypatch.setattr(paper, "now", lambda: NOW)
    monkeypatch.setattr(paper, "read_json", lambda *args, **kwargs: summary)

    def limited(*args, **kwargs):
        pytest.fail("A fresh streamed quote should require no REST book request")

    monkeypatch.setattr(paper.SCANNER_BOOKS, "request", limited)
    row, entry = paper.scan_game(paper.game_state(fixture()), [(EVENT, contract)], {}, book_stream=stream)
    assert row["status"] == "pending_recheck" and row["book_source"] == "websocket"
    assert entry["market"] == MARKET and entry["screened_ev"] >= .05


def test_wcbb_subscribes_pregame_and_caps_rest_fallback_across_workers(monkeypatch, tmp_path):
    contract, summary, _ = inputs(1, .73, ".50")
    upcoming = "wcbb-lsu-vill-2026-11-27"
    upcoming_contract = {**contract, "slug": "aec-" + upcoming,
                         "gameStartTime": (NOW + timedelta(minutes=3)).isoformat(),
                         "marketSides": [{"long": True, "team": {"name": "LSU (W)"}},
                                         {"long": False, "team": {"name": "Villanova (W)"}}]}
    stream = BookStream(None, Event(), request_id="wcbb-anytime")
    monkeypatch.setattr(paper, "now", lambda: NOW)
    monkeypatch.setattr(paper.halftime_paper, "live_inventory", lambda *args, **kwargs: (
        {"WCBB": [(EVENT, contract), (upcoming, upcoming_contract)]}, []))
    monkeypatch.setattr(paper.halftime_paper, "scores", lambda *args: [
        {**fixture(), "id": str(game_id)} for game_id in range(101, 108)])
    monkeypatch.setattr(paper, "read_json", lambda *args, **kwargs: summary)
    requests = []

    def limited(path, fetch, *, fresh_book):
        assert fresh_book is True
        requests.append(path)
        raise paper.BookRateLimited(10)

    monkeypatch.setattr(paper.SCANNER_BOOKS, "request", limited)
    result = paper.scan(tmp_path / "paper.sqlite3", book_stream=stream)
    assert stream.markets == {MARKET, "aec-" + upcoming}
    assert len(requests) == paper.REST_BOOKS_PER_STREAM_SCAN
    statuses = [row["status"] for row in result["games"]]
    assert statuses.count("book_rate_limited") == 4
    assert statuses.count("waiting_for_stream_book") == 3
    assert not result["entries"]


def test_discovers_open_moneyline_before_exchange_marks_game_live(monkeypatch):
    contract, _, _ = inputs(1, .73, ".50")
    delayed = {"slug": EVENT, "live": False, "ended": False, "markets": [contract]}
    finished = {**delayed, "ended": True}
    outside = {**delayed, "markets": [{**contract, "gameStartTime": (NOW - timedelta(days=3)).isoformat()}]}
    start_window = (NOW - timedelta(days=1), NOW + timedelta(days=1))

    def inventory_response(*args, params, **kwargs):
        assert params["tagSlug"] == "wcbb"
        assert "live" not in params
        assert params["startDateMin"] == start_window[0].isoformat()
        assert params["startDateMax"] == start_window[1].isoformat()
        return {"events": [delayed, finished, outside]}

    monkeypatch.setattr(paper.halftime_paper, "read_json", inventory_response)
    listings, live = paper.halftime_paper.live_inventory("WCBB", start_window=start_window)
    assert live == []
    game = {**paper.game_state(fixture()), "listed_moneylines": listings["WCBB"]}
    event, market, assignments = paper.halftime_paper.contract_and_sides(game)
    assert (event["slug"], market["slug"], assignments) == (EVENT, MARKET, {"away": True, "home": False})


def test_scan_persists_once_and_uses_exchange_settlement(monkeypatch, tmp_path):
    path = tmp_path / "wcbb.sqlite3"
    contract, summary, book = inputs(4, .75, ".50")
    monkeypatch.setattr(paper, "now", lambda: NOW)
    monkeypatch.setattr(paper.halftime_paper, "live_inventory",
                        lambda league, *, start_window: ({"WCBB": [(EVENT, contract)]}, []))
    monkeypatch.setattr(paper, "read_json", lambda *args, **kwargs: summary)
    monkeypatch.setattr(paper, "public", lambda *args, **kwargs: {"marketData": book})
    calls = iter([fixture(4), fixture(4, 18)])
    monkeypatch.setattr(paper.halftime_paper, "scores", lambda *args: [next(calls)])
    rejected = paper.scan(path)
    assert rejected["games"][0]["status"] == "state_changed"
    assert rejected["entries"] == []
    monkeypatch.setattr(paper.halftime_paper, "scores", lambda *args: [fixture(4)])
    accepted = paper.scan(path)
    assert accepted["performance"]["open"] == 1
    assert accepted["performance"]["roi"] is None
    repeat = paper.scan(path)
    assert len(repeat["entries"]) == 1
    assert repeat["games"][0]["market_url"] == repeat["entries"][0]["market_url"] == paper.moneyline_url(EVENT, MARKET)
    market = {"slug": MARKET, "status": "MARKET_STATUS_OPEN", "marketSides": [
        {"long": False, "price": "1"}, {"long": True, "price": "0"},
    ]}
    monkeypatch.setattr(paper, "now", lambda: NOW + timedelta(minutes=5))

    def market_result(method, url, *, params, **kwargs):
        assert (method, url) == ("GET", f"{paper.GATEWAY}/v1/markets")
        assert params == {"slug": [MARKET], "limit": 1}
        return {"markets": [market]}

    monkeypatch.setattr(paper, "read_json", market_result)
    paper.settle(path)
    assert paper.read_ledger(path)[0]["settlement"] is None
    market["status"] = "MARKET_STATUS_RESOLVED"
    paper.settle(path)
    settled = paper.scan(path)
    assert settled["performance"]["wins"] == 1
    assert settled["performance"]["open"] == 0
    assert settled["performance"]["roi"] > 0
    assert settled["entries"][0]["result"] == "Won"
    state = ScanState()
    state.wcbb = settled
    assert state.read()["wcbb_anytime"] == settled


def test_settlement_batches_existing_entries_by_exact_market_slug(monkeypatch, tmp_path):
    path = tmp_path / "wcbb.sqlite3"
    picks = {"401": ("duke-unc", True), "402": ("fair-byu", False),
             "403": ("stjohn-rich", True), "404": ("verm-gmsn", True)}
    markets = []
    for game_id, (teams, long) in picks.items():
        event = f"wcbb-{teams}-2026-11-27"
        slug = f"aec-{event}"
        paper.record_entry(path, {"game_id": game_id, "event": event, "market": slug,
                                  "long": long, "recorded_at": NOW.isoformat()})
        markets.append({"slug": slug, "status": "MARKET_STATUS_RESOLVED", "marketSides": [
            {"long": True, "price": "1"}, {"long": False, "price": "0"},
        ]})
    markets[2]["marketSides"][1]["price"] = "1"  # Complementary payouts are required.
    markets[3]["slug"] = "aec-wcbb-verm-gmsn-2026-11-28"
    requests = []

    def market_result(method, url, *, params, **kwargs):
        requests.append(params["slug"])
        return {"markets": list(reversed(markets))}

    monkeypatch.setattr(paper, "read_json", market_result)
    paper.settle(path)
    assert len(requests) == 1
    assert len(requests[0]) == 4
    entries = paper.read_ledger(path)
    assert {row["game_id"]: row["settlement"] for row in entries} == {
        "401": "1", "402": "0", "403": None, "404": None,
    }
    paper.settle(path)
    assert requests[1] == [requests[0][2], requests[0][3]]
    assert paper.read_ledger(path) == entries


def test_live_scanner_rechecks_state_and_records_a_single_capped_order(monkeypatch, tmp_path):
    path = tmp_path / "paper.sqlite3"
    live_path = tmp_path / "live.sqlite3"
    contract, summary, book = inputs(4, .75, ".50")
    monkeypatch.setattr(paper, "now", lambda: NOW)
    monkeypatch.setattr(pro_halftime_orders, "now", lambda: NOW)
    monkeypatch.setattr(paper.halftime_paper, "live_inventory",
                        lambda league, *, start_window: ({"WCBB": [(EVENT, contract)]}, []))
    monkeypatch.setattr(paper, "read_json", lambda *args, **kwargs: summary)
    monkeypatch.setattr(paper.SCANNER_BOOKS, "read", lambda *args, **kwargs: book)
    calls = iter([fixture(4), fixture(4, 18), fixture(4), fixture(4), fixture(4)])
    monkeypatch.setattr(paper.halftime_paper, "scores", lambda *args: [next(calls)])
    submitted = []

    def submit(payload):
        submitted.append(payload)
        return {"id": "order-123"}

    def retrieve(order_id):
        assert order_id == "order-123"
        return {"order": {"id": order_id, "marketSlug": MARKET, "state": "ORDER_STATE_FILLED", "cumQuantity": 9}}

    def client():
        return nullcontext(SimpleNamespace(orders=SimpleNamespace(create=submit, retrieve=retrieve)))

    ledger = pro_halftime_orders.AutomaticOrders(live_path, client, max_book_age=paper.BOOK_MAX_AGE)
    changed = paper.scan(path, live_orders=ledger)
    assert changed["games"][0]["status"] == "state_changed"
    assert changed["orders"] == [] and not submitted

    accepted = paper.scan(path, live_orders=ledger)
    assert accepted["live_ordering"] is True
    assert accepted["games"][0]["status"] == "live_order_recorded"
    assert accepted["entries"] == [] and not path.exists()
    entry, = accepted["orders"]
    assert entry["state"] == "submitted"
    assert "filled_quantity" not in entry
    assert entry["market_url"] == paper.moneyline_url(EVENT, MARKET)
    assert .60 < entry["probability"] and entry["screened_ev"] >= .05
    assert float(entry["cost"]) <= 5 and entry["quantity"] == 9
    assert submitted == [{"marketSlug": MARKET, "type": "ORDER_TYPE_LIMIT",
                          "price": {"value": "0.50", "currency": "USD"}, "quantity": 9,
                          "tif": "TIME_IN_FORCE_FILL_OR_KILL", "intent": "ORDER_INTENT_BUY_SHORT",
                          "manualOrderIndicator": "MANUAL_ORDER_INDICATOR_AUTOMATIC", "synchronousExecution": True}]

    restarted = pro_halftime_orders.AutomaticOrders(live_path, client)
    restarted.refresh()
    repeat = paper.scan(path, live_orders=restarted)
    assert repeat["games"][0]["status"] == "live_order_already_attempted"
    assert repeat["orders"][0]["state"] == "ORDER_STATE_FILLED"
    assert repeat["orders"][0]["filled_quantity"] == 9
    assert len(submitted) == 1


def test_wcbb_refreshes_expired_order_only_with_new_qualifying_quote_and_zero_confirmed_fills(monkeypatch, tmp_path):
    paper_path = tmp_path / "paper.sqlite3"
    path = tmp_path / "wcbb.sqlite3"
    contract, summary, book = inputs(1, .75, ".50")
    moment = NOW
    monkeypatch.setattr(paper, "now", lambda: moment)
    monkeypatch.setattr(pro_halftime_orders, "now", lambda: moment)
    monkeypatch.setattr(paper.halftime_paper, "live_inventory",
                        lambda league, *, start_window: ({"WCBB": [(EVENT, contract)]}, []))
    monkeypatch.setattr(paper, "read_json", lambda *args, **kwargs: summary)
    monkeypatch.setattr(paper.halftime_paper, "scores", lambda *args: [fixture()])
    monkeypatch.setattr(paper.SCANNER_BOOKS, "read", lambda *args, **kwargs: book)
    fresh_books, submissions, lookups = [], [], []

    def refresh_book(path, fetch, *, fresh_book):
        assert fresh_book is True and MARKET in path
        fresh_books.append(book["transactTime"])
        return book

    def submit(payload):
        submissions.append(payload)
        return {"id": f"order-{len(submissions)}", "executions": [{"order": {
            "state": pro_halftime_orders.EXPIRED, "cumQuantity": 0,
        }}]}

    exchange_quantity = 1

    def retrieve(order_id):
        lookups.append(order_id)
        return {"order": {"id": order_id, "marketSlug": MARKET, "state": pro_halftime_orders.EXPIRED,
                          "cumQuantity": exchange_quantity}}

    monkeypatch.setattr(paper.SCANNER_BOOKS, "request", refresh_book)

    def client():
        return nullcontext(SimpleNamespace(orders=SimpleNamespace(create=submit, retrieve=retrieve)))

    original = pro_halftime_orders.AutomaticOrders(path, client)
    first = paper.scan(paper_path, live_orders=original)
    assert first["orders"][0]["filled_quantity"] == 0

    restarted = pro_halftime_orders.AutomaticOrders(path, client, max_book_age=paper.BOOK_MAX_AGE, retry_expired=True)
    moment += timedelta(seconds=1)
    stale = paper.scan(paper_path, live_orders=restarted)
    assert stale["games"][0]["status"] == "waiting_for_retry_quote"

    book["transactTime"] = moment.isoformat()
    book["bids"][0]["px"]["value"] = ".25"
    failed_screen = paper.scan(paper_path, live_orders=restarted)
    assert failed_screen["games"][0]["status"] == "below_ev"
    assert not lookups and len(submissions) == 1

    book["bids"][0]["px"]["value"] = ".47"
    unconfirmed = paper.scan(paper_path, live_orders=restarted)
    assert unconfirmed["games"][0]["status"] == "live_order_retry_unconfirmed"
    assert len(submissions) == 1

    exchange_quantity = 0
    accepted = paper.scan(paper_path, live_orders=restarted)
    retry, prior = accepted["orders"]
    assert (retry["order_id"], prior["order_id"]) == ("order-2", "order-1")
    assert retry["attempt_number"] == 2 and float(retry["cost"]) <= 5
    assert submissions[1]["price"]["value"] == "0.47"
    assert submissions[1]["tif"] == "TIME_IN_FORCE_FILL_OR_KILL"
    assert fresh_books[-1] == moment.isoformat()

    repeat = paper.scan(paper_path, live_orders=restarted)
    assert repeat["games"][0]["status"] == "live_order_already_attempted"
    assert lookups == ["order-1", "order-1"] and len(submissions) == 2


def test_concurrent_retries_reserve_only_one_replacement(monkeypatch, tmp_path):
    moment = NOW
    monkeypatch.setattr(pro_halftime_orders, "now", lambda: moment)
    checked = Barrier(2, timeout=5)
    submissions = []

    def submit(payload):
        submissions.append(payload)
        return {"id": f"order-{len(submissions)}", "executions": [{"order": {
            "state": pro_halftime_orders.EXPIRED, "cumQuantity": 0,
        }}]}

    def retrieve(order_id):
        checked.wait()
        return {"order": {"id": order_id, "marketSlug": MARKET, "state": pro_halftime_orders.EXPIRED,
                          "cumQuantity": 0}}

    def client():
        return nullcontext(SimpleNamespace(orders=SimpleNamespace(create=submit, retrieve=retrieve)))

    ledger = pro_halftime_orders.AutomaticOrders(tmp_path / "orders.sqlite3", client, retry_expired=True)

    def make_entry():
        return {"league": paper.LEAGUE, "event": EVENT, "official_game_id": "40123",
                "recorded_at": moment.isoformat(), "quantity": 9,
                "quote": {"quote_utc": moment.isoformat(), "total_estimate": "4.84"}}

    payload = {"marketSlug": MARKET}
    assert ledger.place(make_entry(), payload)["status"] == "live_order_recorded"
    moment += timedelta(seconds=1)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: ledger.place(make_entry(), payload, retry_for="order-1"), range(2)))
    assert {result["status"] for result in results} == {"live_order_already_attempted", "live_order_recorded"}
    assert len(submissions) == len(ledger.recent()) == 2


@pytest.mark.parametrize("auth_ok", [True, False])
def test_dashboard_runner_enables_only_wcbb_live_after_auth_check(monkeypatch, tmp_path, auth_ok):
    for name in (run_research.KEY_ID_VAR, run_research.SECRET_VAR, "POLYMARKET_KEY_ID", "POLYMARKET_SECRET_KEY",
                 "PG_DISABLE_AUTO_REFRESH", "PG_ENABLE_PRO_HALFTIME_PAPER", "PARLAY_GUARD_US_ORDERS"):
        monkeypatch.delenv(name, raising=False)
    secret = b64encode(bytes(32)).decode()
    env = tmp_path / ".env"
    env.write_text(f"{run_research.KEY_ID_VAR}=test-wcbb\n{run_research.SECRET_VAR}={secret}\n")
    monkeypatch.setattr(env_config, "PROJECT_ENV", env)
    monkeypatch.setattr(run_research, "ROOT", tmp_path)

    def balances():
        if not auth_ok:
            raise OSError("mocked connection failure")
        return {"balances": []}

    def sdk_client(*, key_id, secret_key, timeout, max_retries):
        assert (key_id, secret_key, max_retries) == ("test-wcbb", secret, 0)
        return nullcontext(SimpleNamespace(account=SimpleNamespace(balances=balances)))

    app = object()
    scanner = ScanState()
    started = []
    monkeypatch.setitem(sys.modules, "polymarket_us", SimpleNamespace(PolymarketUS=sdk_client))
    monkeypatch.setitem(sys.modules, "src.api", SimpleNamespace(app=app, inplay_scanner=scanner))
    monkeypatch.setattr(run_research.uvicorn, "run", lambda value, **_: started.append(value))
    monkeypatch.setattr(sys, "argv", ["run_research", "--live-wcbb"])
    if auth_ok:
        run_research.main()
        assert started == [app]
        assert scanner.wcbb_orders.path == tmp_path / paper.LIVE_DB_PATH
        assert scanner.wcbb_orders.retry_expired is True
        assert scanner.wcbb_books.request_id == "wcbb-anytime"
        assert scanner.read()["wcbb_anytime"]["live_ordering"]
    else:
        with pytest.raises(SystemExit) as error:
            run_research.main()
        assert error.value.code == 2
        assert not started and scanner.wcbb_orders is None
    assert scanner.pro_halftime_orders is None
