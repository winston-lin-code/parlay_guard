import csv
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from src import nba_moneyline_shadow as study


NOW = datetime(2026, 11, 28, 1, tzinfo=timezone.utc)
EVENT = "nba-pho-hou-2026-11-27"
MARKET = "aec-" + EVENT


def test_shadow_study_keeps_same_limit_for_both_sizes_and_settles_once(monkeypatch, tmp_path):
    moment = NOW
    tip = NOW - timedelta(minutes=40)
    event = {"id": "40123", "date": tip.isoformat(), "season": {"type": 2}, "competitions": [{
        "status": {"period": 2, "clock": 222, "displayClock": "3:42",
                   "type": {"state": "in", "name": "STATUS_IN_PROGRESS"}},
        "competitors": [{"homeAway": "home", "team": {"id": "10", "abbreviation": "HOU"}, "score": "58"},
                        {"homeAway": "away", "team": {"id": "20", "abbreviation": "PHX"}, "score": "51"}],
    }]}
    contract = {"slug": MARKET, "question": "Suns vs Rockets", "marketType": "moneyline",
                "sportsMarketType": "basketball_team_full_game_winner", "minimumTradeQty": "1",
                "orderPriceMinTickSize": ".01", "feeCoefficient": ".11", "gameStartTime": tip.isoformat(),
                "description": "Overtime is included.", "active": True, "closed": False, "marketSides": [
                    {"long": True, "tradable": True, "team": {"abbreviation": "PHX"}},
                    {"long": False, "tradable": True, "team": {"abbreviation": "HOU"}},
                ]}
    summary = {"header": {"id": "40123"}, "plays": [{"id": "play-1", "homeScore": 58, "awayScore": 51,
               "wallclock": (NOW - timedelta(seconds=20)).isoformat(), "period": {"number": 2}}],
               "winprobability": [{"playId": "play-1", "homeWinPercentage": .76}]}
    reads = []

    def book(slug):
        assert slug == MARKET
        reads.append(moment)
        bids = [{"px": {"value": ".50", "currency": "USD"}, "qty": "20" if len(reads) == 1 else "4"}]
        if len(reads) > 1:
            bids.append({"px": {"value": ".45", "currency": "USD"}, "qty": "30"})
        return {"marketSlug": slug, "transactTime": moment.isoformat(), "state": "MARKET_STATE_OPEN", "bids": bids}

    def delay(seconds):
        nonlocal moment
        moment += timedelta(seconds=seconds)

    monkeypatch.setattr(study, "now", lambda: moment)
    monkeypatch.setattr(study, "sleep", delay)
    monkeypatch.setattr(study, "fresh_book", book)
    monkeypatch.setattr(study.halftime, "read_json", lambda *args, **kwargs: summary)
    monkeypatch.setattr(study, "scoreboard", lambda *args: [event])
    monkeypatch.setattr(study.halftime, "live_inventory", lambda *args, **kwargs: ({"NBA": [(EVENT, contract)]}, []))
    ledger = study.Ledger(tmp_path)
    assert study.scan(ledger, 2) == (1, 2, [])
    assert study.scan(study.Ledger(tmp_path), 2) == (1, 0, [])
    assert reads == [NOW, NOW + timedelta(seconds=2)]

    market = {"slug": MARKET, "status": "MARKET_STATUS_RESOLVED",
              "marketSides": [{"long": True, "price": "0"}, {"long": False, "price": "1"}]}
    monkeypatch.setattr(study, "settlement_market_batches", lambda _: [[market]])
    ledger.settle()
    study.report(ledger)
    with (tmp_path / "signals.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4 and {row["limit"] for row in rows} == {"0.50"}
    assert all(row["observed_delay_seconds"] == "2.0" for row in rows)
    assert {row["net_pnl"] for row in rows if row["size"] == "full"} == {"0"}
    assert {row["net_pnl"] for row in rows if row["size"] == "half_shares"} == {"1.85"}
    with (tmp_path / "summary.csv").open(newline="") as handle:
        stats = list(csv.DictReader(handle))
    assert {row["roi_on_settled_cost"] for row in stats if row["size"] == "full"} == {""}
    assert all(Decimal(row["roi_on_settled_cost"]) > Decimal(".86") for row in stats if row["size"] == "half_shares")


def test_unobserved_book_is_excluded_from_fill_rate_and_pnl():
    entries = [{"window": "anytime", "full_quantity": 6, "half_shares_quantity": 3,
                "payout": "1", "execution": {"reason": "stale_or_wrong_book"}}]
    rows = list(study.trade_rows(entries))
    stats = study.performance(rows, "anytime", "full")
    assert stats["unobserved"] == 1 and stats["fill_rate_observed"] is None
    assert stats["pnl_per_evaluated_signal"] is None
