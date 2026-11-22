"""Build audited pregame features for upcoming Polymarket NBA props and predict.

Run AFTER training and a fresh Polymarket scan:
    python train_player_props.py
    python polymarket_nba_ev.py --pages 2 --max-games 5
    python predict_current_props.py --audit-only
    python predict_current_props.py

Inputs:
  models/player_prop_models_2026_27.joblib
  data/processed/player_training_data_v2.csv  (must contain RECENT completed 2026-27 games)
  data/polymarket_nba_ev.csv  (fresh future game markets)

Outputs:
  data/current_projections.csv
  data/current_player_features.csv

Conservative guards:
- Audits reconstructed features against actual historical pregame feature columns.
- Uses only previous completed game dates (strictly before each matchup's local date).
- Matches players to a team in the Polymarket NBA event slug.
- Requires recent current-season history; skips stale/ambiguous/unknown players.
- Assumes NBA event slug format is nba-AWAY-HOME-yyyy-mm-dd; verify externally.
- Does not check injuries or certainty of player participation; EV is exploratory.
"""
from __future__ import annotations

import argparse
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .player_prop_common import (
    FEATURES, TARGETS, TEAM_ID_TO_ABBR, audit_feature_parity,
    build_feature_row, name_key, prepare_history, season_of_date,
)

MODEL = Path("models/player_prop_models_2026_27.joblib")
HISTORY = Path("data/processed/player_training_data_v2.csv")
MARKETS = Path("data/polymarket_nba_ev.csv")
OUTPUT = Path("data/current_projections.csv")
FEATURE_OUTPUT = Path("data/current_player_features.csv")
SLUG = re.compile(r"^nba-(?P<away>[a-z]{2,4})-(?P<home>[a-z]{2,4})-\d{4}-\d{2}-\d{2}$")


def parse_market_games(markets: pd.DataFrame, now: pd.Timestamp) -> pd.DataFrame:
    necessary = {"event_slug", "game_start_utc", "player_name", "stat"}
    missing = necessary - set(markets.columns)
    if missing:
        raise ValueError(f"Market CSV lacks {sorted(missing)}; rerun updated polymarket_nba_ev.py")
    markets = markets[markets["stat"].astype(str).isin(TARGETS)].copy()
    markets["game_start_utc"] = pd.to_datetime(markets["game_start_utc"], utc=True, errors="coerce")
    markets = markets[markets["game_start_utc"].notna() & (markets["game_start_utc"] > now)].copy()
    markets["PLAYER_KEY"] = markets["player_name"].map(name_key)
    markets = markets.drop_duplicates(["event_slug", "game_start_utc", "PLAYER_KEY"])
    return markets.sort_values(["game_start_utc", "PLAYER_KEY"]).reset_index(drop=True)


def infer_context(market_row, previous: pd.DataFrame, min_season_games: int, age_limit: int):
    match = SLUG.fullmatch(str(market_row.event_slug))
    if not match:
        return None, "unknown_event_slug"
    away, home = match.group("away"), match.group("home")
    game_start = market_row.game_start_utc
    game_day = game_start.tz_convert("America/New_York").tz_localize(None).normalize()
    season = season_of_date(game_day)
    prior = previous[previous["GAME_DATE"] < game_day].copy()
    if prior.empty:
        return None, "no_prior_games"
    # The processed training file excludes earliest appearances, so at least
    # 20 *retained* prior games are required for a faithful 20-game window.
    if len(prior) < 20:
        return None, "insufficient_retained_history_for_20_game_window"
    this_season = prior[prior["SEASON"].astype(str) == season]
    if len(this_season) < min_season_games:
        return None, "insufficient_current_season_games"
    latest_game = prior.sort_values(["GAME_DATE", "GAME_ID"] if "GAME_ID" in prior.columns else ["GAME_DATE"]).iloc[-1]
    if (game_day - latest_game["GAME_DATE"]).days > age_limit:
        return None, "history_too_stale"
    try:
        team_id = int(latest_game["TEAM_ID"])
    except (TypeError, ValueError, OverflowError):
        return None, "unknown_team_id"
    team = TEAM_ID_TO_ABBR.get(team_id)
    if team not in (away, home):
        return None, "team_not_in_event"
    row = build_feature_row(prior, target_day=game_day, season=season,
                            is_home=int(team == home), rest_offset=-1)
    row.update({
        "player_name": str(market_row.player_name),
        "PLAYER_ID": latest_game["PLAYER_ID"],
        "team": team.upper(),
        "event_slug": str(market_row.event_slug),
        "game_start_utc": game_start.isoformat(),
        "last_completed_game_date": latest_game["GAME_DATE"].date().isoformat(),
        "current_season_games": row["CURRENT_SEASON_GAMES_PLAYED"],
        "feature_origin": "engineered_history_anchored_counters_and_event_slug",
    })
    return row, None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, default=HISTORY)
    parser.add_argument("--markets", type=Path, default=MARKETS)
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--audit-only", action="store_true", help="Check training/forecast feature parity and stop")
    parser.add_argument("--min-current-games", type=int, default=3)
    parser.add_argument("--max-history-age-days", type=int, default=7)
    args = parser.parse_args(argv)
    for path in ([args.history, args.model] if args.audit_only else [args.history, args.markets, args.model]):
        if not path.exists():
            parser.error(f"Missing {path}")
    if args.min_current_games < 1 or args.max_history_age_days < 1:
        parser.error("min-current-games and max-history-age-days must be positive")

    # Trusted local artifact only; joblib uses pickle under the hood.
    bundle = joblib.load(args.model)
    if bundle.get("features") != FEATURES or set(bundle.get("models", {})) != set(TARGETS):
        raise ValueError("Saved player model bundle has different feature definitions; retrain first")
    print("Loaded four player-prop regressors (not the team Elo/win models).")
    history = prepare_history(pd.read_csv(args.history, low_memory=False))
    print(f"History: {len(history):,} player-games | latest date: {history['GAME_DATE'].max().date()}")
    print(f"Seasons: {history.SEASON.value_counts().sort_index().to_dict()}")
    # An offset around 20 is a sign the feature-ready dataset omitted early
    # appearances, while preserving original career pregame counters.
    first_rows = history.sort_values(["PLAYER_ID", "GAME_DATE"]).groupby("PLAYER_ID", sort=False).first()
    first_counts = pd.to_numeric(first_rows["GAMES_PLAYED_BEFORE"], errors="coerce").dropna().astype(int)
    print(f"First retained appearance: GAMES_PLAYED_BEFORE distribution "
          f"{dict(Counter(first_counts).most_common(6))}")
    passed, issues, n = audit_feature_parity(history)
    print(f"Feature parity audit: {'PASS' if passed else 'FAIL'} ({n} recent pregame rows checked)")
    if issues:
        for issue in issues:
            print("  ", issue)
    if not passed:
        raise SystemExit("STOP: Current feature calculations do not match training features. "
                         "Share the feature-engineering script so we can align formulas. "
                         "No current projections were written.")
    if args.audit_only:
        print("Audit successful; no projections requested.")
        return 0

    now = pd.Timestamp(datetime.now(timezone.utc))
    markets = parse_market_games(pd.read_csv(args.markets, low_memory=False), now)
    if markets.empty:
        raise SystemExit("No upcoming player markets in the market CSV. Rerun Polymarket scanner first.")
    print(f"Unique future player/game pairs in market CSV: {len(markets)}")
    # Audit player-name collisions. Historical player IDs are authoritative;
    # don't guess which person a matching name refers to.
    by_key = {key: grp for key, grp in history.groupby("PLAYER_KEY", sort=False)}
    rejected = Counter()
    feature_rows = []
    for m in markets.itertuples(index=False):
        candidates = by_key.get(m.PLAYER_KEY)
        if candidates is None:
            rejected["player_missing_from_history"] += 1
            continue
        ids = candidates["PLAYER_ID"].dropna().unique()
        if len(ids) != 1:
            rejected["ambiguous_player_name"] += 1
            continue
        row, reason = infer_context(m, candidates, args.min_current_games, args.max_history_age_days)
        if reason:
            rejected[reason] += 1
            continue
        feature_rows.append(row)
    if not feature_rows:
        raise SystemExit(f"No safe current feature rows. Rejection reasons: {dict(rejected)}. "
                         "Check latest 2026-27 player history and correct team identifiers.")
    feature_df = pd.DataFrame(feature_rows)
    if feature_df["player_name"].map(name_key).duplicated().any():
        raise SystemExit("Same player appears in more than one future game. Current scanner joins by name/stat only; stop to avoid mixing games.")
    # Explicitly reject nonnumeric or infinite features. Models can handle NaN.
    for feature in FEATURES:
        feature_df[feature] = pd.to_numeric(feature_df[feature], errors="coerce").replace([np.inf, -np.inf], np.nan)
    if feature_df[FEATURES].notna().sum(axis=1).min() < 60:
        raise SystemExit("One or more players have too many missing features; no projections written.")
    generated = now.isoformat()
    predictions = []
    for stat in TARGETS:
        model = bundle["models"][stat]
        output = np.maximum(0.0, model.predict(feature_df[FEATURES]))
        for i, pred in enumerate(output):
            row = feature_df.iloc[i]
            predictions.append({
                "player_name": row["player_name"], "stat": stat,
                "prediction": round(float(pred), 4), "as_of_utc": generated,
                "event_slug": row["event_slug"], "game_start_utc": row["game_start_utc"],
                "last_completed_game_date": row["last_completed_game_date"],
                "team": row["team"], "feature_audit_passed": True,
                "player_availability_verified": False,
                "model_training_end_season": "2025-26",
            })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    feature_df.to_csv(FEATURE_OUTPUT, index=False)
    pd.DataFrame(predictions).to_csv(args.output, index=False)
    print(f"Generated projections: {len(predictions)} for {len(feature_df)} players")
    print(f"Skipped player/game pairs: {dict(rejected)}")
    print(f"Saved: {FEATURE_OUTPUT}")
    print(f"Saved: {args.output}")
    print("WARNING: Event slug order assumes away-home, and player availability/injuries are NOT verified.")
    print("WARNING: Do not treat estimated EV as actionable until these assumptions and calibration are validated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
