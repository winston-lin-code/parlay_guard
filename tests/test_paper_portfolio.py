"""Offline tests for simulated bankroll, NBA box-score verification and metrics."""
from datetime import datetime, timedelta, timezone

import pandas as pd

from src.paper_auto_execution import on_live_refresh
from src.paper_portfolio import portfolio_report, settle_pending


def _snapshot(now, *, away="NYK", home="BOS", outcome_price=.4):
    return {
        "updated_at_utc": (now - timedelta(seconds=3)).isoformat(),
        "games": [{
            "slug": f"nba-{away.lower()}-{home.lower()}-{now.date().isoformat()}",
            "away": away, "home": home,
            "tipoff_utc": (now + timedelta(hours=1)).isoformat(),
            "winner_forecast": {
                "predicted_winner": home, "home_probability": .75,
                "forecast_generated_at_utc": (now - timedelta(minutes=5)).isoformat(),
            },
            "moneylines": {home: {
                "ask": outcome_price, "shares": 100, "token_id": "123456789",
                "observed_at_utc": (now - timedelta(seconds=5)).isoformat(),
            }},
        }],
    }


def _team_results(path, *, winner="BOS"):
    path.parent.mkdir(parents=True, exist_ok=True)
    bos_win = winner == "BOS"
    pd.DataFrame([
        {"GAME_ID": "0022600001", "GAME_DATE": "2026-11-21",
         "TEAM_ABBREVIATION": "BOS", "PTS": 110 if bos_win else 90,
         "WL": "W" if bos_win else "L", "MATCHUP": "BOS vs. NYK"},
        {"GAME_ID": "0022600001", "GAME_DATE": "2026-11-21",
         "TEAM_ABBREVIATION": "NYK", "PTS": 90 if bos_win else 110,
         "WL": "L" if bos_win else "W", "MATCHUP": "NYK @ BOS"},
    ]).to_csv(path, index=False)


def test_starts_empty_and_tracks_open_cost(tmp_path):
    report = portfolio_report(tmp_path)
    assert report["status"] == "NOT_STARTED"
    assert report["summary"]["available_cash"] == 100
    now = datetime(2026, 11, 21, 21, tzinfo=timezone.utc)
    assert len(on_live_refresh(_snapshot(now), tmp_path, at=now)["assumed_fills"]) == 1
    result = portfolio_report(tmp_path)
    assert result["summary"]["starting_bankroll"] == 100
    assert result["summary"]["available_cash"] == 95
    assert result["summary"]["reserved_open_stakes"] == 5
    assert result["summary"]["book_value_paper_equity"] == 100
    assert result["summary"]["open_positions"] == 1
    assert result["summary"]["realized_paper_pnl"] == 0
    assert result["positions"][0]["status"] == "PENDING_NBA_VERIFICATION"


def test_verified_pair_settles_once_and_reconciles_cash(tmp_path):
    now = datetime(2026, 11, 21, 21, tzinfo=timezone.utc)
    on_live_refresh(_snapshot(now), tmp_path, at=now)
    later = now + timedelta(days=1)
    assert settle_pending(tmp_path, at=later)["pending"] == 1
    path = tmp_path / "data/nba_2026_27_latest_raw.csv"
    _team_results(path)
    first = settle_pending(tmp_path, at=later)
    assert first["newly_settled"] == 1
    assert settle_pending(tmp_path, at=later)["newly_settled"] == 0
    report = portfolio_report(tmp_path)
    stats = report["summary"]
    assert stats["open_positions"] == 0
    assert stats["settled_trades"] == 1
    assert stats["wins"] == 1
    assert stats["available_cash"] == 107.20
    assert stats["realized_paper_pnl"] == 7.20
    assert stats["settled_paper_roi"] == 1.44
    assert report["positions"][0]["nba_game_id_verified"] == "0022600001"
    scenario = report["execution_cost_scenarios"]
    assert [s["ask_premium_cents"] for s in scenario] == [0, 1, 2]
    assert scenario[0]["simulated_profit"] > scenario[1]["simulated_profit"] > scenario[2]["simulated_profit"]


def test_invalid_or_unpaired_team_result_remains_pending(tmp_path):
    now = datetime(2026, 11, 21, 21, tzinfo=timezone.utc)
    on_live_refresh(_snapshot(now), tmp_path, at=now)
    path = tmp_path / "data/nba_2026_27_latest_raw.csv"
    _team_results(path)
    # One team's row is not an authoritative completed paired game.
    pd.read_csv(path).head(1).to_csv(path, index=False)
    settlement = settle_pending(tmp_path, at=now + timedelta(days=1))
    assert settlement["newly_settled"] == 0
    assert settlement["pending"] == 1
    assert portfolio_report(tmp_path)["positions"][0]["status"] == "PENDING_NBA_VERIFICATION"


def test_loss_and_cash_reconciliation(tmp_path):
    now = datetime(2026, 11, 21, 21, tzinfo=timezone.utc)
    on_live_refresh(_snapshot(now), tmp_path, at=now)
    _team_results(tmp_path / "data/nba_2026_27_latest_raw.csv", winner="NYK")
    settle_pending(tmp_path, at=now + timedelta(days=1))
    stats = portfolio_report(tmp_path)["summary"]
    assert stats["losses"] == 1
    assert stats["realized_paper_pnl"] == -5
    assert stats["available_cash"] == 95
    assert stats["book_value_paper_equity"] == 95
    assert stats["max_realized_drawdown"] == 5


def test_prevent_trades_exceeding_remaining_paper_cash(tmp_path, monkeypatch):
    import src.paper_portfolio as paper
    monkeypatch.setattr(paper, "STARTING_BANKROLL_CENTS", 500)
    now = datetime(2026, 11, 21, 21, tzinfo=timezone.utc)
    data = _snapshot(now)
    other = _snapshot(now, away="CHI", home="DET")["games"][0]
    data["games"].append(other)
    result = on_live_refresh(data, tmp_path, at=now)
    assert len(result["assumed_fills"]) == 1
    assert set(result["skipped"].values()) == {"insufficient_paper_cash"}
    assert portfolio_report(tmp_path)["summary"]["available_cash"] == 0
