from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from src import paper_halftime as paper


NOW = datetime(2026, 11, 26, 2, tzinfo=timezone.utc)


@pytest.fixture
def path(tmp_path):
    file = tmp_path / "paper.sqlite3"
    paper.init_account(file)
    return file


def pick(id=1, **changes):
    slug = f"nba-ny-bos-2026-11-{id}"
    return {"event": slug, "market_slug": f"aec-{slug}", "league": "NBA",
            "fixture_id": str(id), "phase": "halftime", "long": True, "title": "Knicks @ Celtics",
            "side": "Knicks", "score_text": "NY 50 · BOS 48", "probability": .68,
            "screened_ev": .26, "ask": .48, "cost": .50, "quote_utc": NOW.isoformat(),
            "url": "https://polymarket.us/sports/nba/test", **changes}


def scan(*picks, age=0):
    return {"updated_utc": (NOW - timedelta(seconds=age)).isoformat(), "picks": list(picks)}


def totals(path):
    with paper.ledger(path) as db:
        return paper.account_totals(db)


@pytest.mark.parametrize("changes", [
    {"probability": .599},
    {"cost": .61},  # Stays positive, but falls below 5% after the haircut.
    {"quote_utc": (NOW - timedelta(seconds=6)).isoformat()},
    {"phase": "live"},
    {"long": None},
    {"market_slug": "aec-nba-other"},
])
def test_record_scan_revalidates_rule_and_quote(path, changes):
    assert paper.record_scan(path, scan(pick(**changes)), now=NOW) == []
    assert totals(path)["available_cents"] == 10_000


def test_stale_scan_is_skipped_even_if_quote_is_fresh(path):
    assert paper.record_scan(path, scan(pick(), age=6), now=NOW) == []


def test_paper_entries_are_atomic_and_deduplicate_even_across_process_connections(path):
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: paper.record_scan(path, scan(pick()), now=NOW), range(8)))
    assert sum(map(len, results)) == 1
    alt_event = "nba-nyk-bos-2026-11-01"
    assert paper.record_scan(path, scan(pick(event=alt_event, market_slug=f"aec-{alt_event}")), now=NOW) == []
    assert totals(path) == {"bets": 1, "pending": 1, "available_cents": 9800,
                            "pending_stake_cents": 200, "realized_cents": 0}


def test_available_cash_limits_paper_exposure(path):
    assert len(paper.record_scan(path, scan(*(pick(i) for i in range(1, 52))), now=NOW)) == 50
    assert totals(path)["available_cents"] == 0


def test_settles_long_short_and_partial_and_keeps_unresolved_open(path, monkeypatch):
    bets = [pick(1), pick(2, long=False), pick(3), pick(4)]
    assert len(paper.record_scan(path, scan(*bets), now=NOW)) == 4
    settlements = dict(zip((row["market_slug"] for row in bets), [Decimal("1"), Decimal("0"), Decimal(".35"), None]))
    monkeypatch.setattr(paper, "settlement_price", settlements.get)
    changed, failures = paper.settle_pending(path)
    assert failures == 0
    assert [net for _, net in changed] == [200, 200, -60]
    assert totals(path) == {"bets": 4, "pending": 1, "available_cents": 10140,
                            "pending_stake_cents": 200, "realized_cents": 340}
    assert paper.settle_pending(path) == ([], 0)


def test_settlement_uses_public_payout_and_never_guesses_wrong_contract(monkeypatch):
    slug = "aec-nba-ny-bos-2026-11-25"
    class Response:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"slug": slug, "settlement": 0}

    response = Response()
    monkeypatch.setattr(paper.requests, "get", lambda url, **kwargs: response)
    assert paper.settlement_price(slug) == Decimal(0)
    assert paper.settlement_price("aec-nba-bkn-ny-2026-11-25") is None
    response.status_code = 404
    assert paper.settlement_price(slug) is None


def test_status_and_csv_use_the_saved_ledger(path, capsys, tmp_path):
    paper.record_scan(path, scan(pick()), now=NOW)
    paper.print_status(path)
    assert "Paper cash $98.00" in capsys.readouterr().out
    destination = tmp_path / "paper.csv"
    paper.export_csv(path, destination)
    assert "Knicks @ Celtics" in destination.read_text()
    with pytest.raises(FileExistsError):
        paper.export_csv(path, destination)
