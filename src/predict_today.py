"""Make and save pre-tipoff NBA game predictions using the FROZEN Parlay Guard model.

From the project root (PowerShell):
    python -u -m src.predict_today
    python -u -m src.predict_today --date 2026-11-18

Requires:
  - data/nba_games_updated.csv (historical COMPLETED team games through 2025-26)
  - models/elo_final_holdout_model.joblib (from one-time final validation)
  - internet for today's NBA CDN scoreboard and current-season NBA Stats logs

If NBA Stats blocks the current-season download, provide a previously downloaded
LeagueGameFinder CSV from the 2026-27 *regular season*:
    python -u -m src.predict_today --season-csv data/nba_2026_27_export.csv

IMPORTANT: Current-season CSV must include ALL completed regular-season games
before prediction date, not just the most recent 10 games. Elo is replayed from
historical results with the SAME rules as src.elo; rolling stats include previous
seasons, just as in the trained feature pipeline.

Only not-yet-started matchups are predicted. Pre-tipoff predictions are saved with
an immutable UTC timestamp for honest future assessment. The model is NOT refitted.
"""

from __future__ import annotations

import argparse
from datetime import datetime, date, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd

from .common import DATA, MODELS
from .elo import HOME_ADVANTAGE, INITIAL_ELO, K_FACTOR, OFFSEASON_REGRESSION

ET = ZoneInfo("America/New_York")

# ESPN team abbreviations that differ from NBA Stats historical abbreviations.
ESPN_TO_NBA = {"GS": "GSW", "NO": "NOP", "UTAH": "UTA", "WSH": "WAS", "NY": "NYK", "SA": "SAS", "PHO": "PHX"}

def normalize_schedule_team(abbrev: str) -> str:
    code = str(abbrev).upper().strip()
    return ESPN_TO_NBA.get(code, code)

EXPECTED_INPUTS = [
    "HOME_WIN_PCT_LAST_5", "HOME_WIN_PCT_LAST_10", "HOME_DAYS_REST",
    "HOME_POINT_DIFF_LAST_10", "AWAY_WIN_PCT_LAST_5",
    "AWAY_WIN_PCT_LAST_10", "AWAY_DAYS_REST",
    "AWAY_POINT_DIFF_LAST_10", "ELO_DIFF",
]


def as_date(value: str | date) -> date:
    return date.fromisoformat(str(value)[:10])


def fetch_scoreboard(target: date) -> pd.DataFrame:
    """NBA's daily live CDN feed; never use live/in-progress games as pregame picks."""
    try:
        from nba_api.live.nba.endpoints import scoreboard
        raw = scoreboard.ScoreBoard().get_dict()
    except Exception as e:
        raise RuntimeError(
            "Could not retrieve NBA live scoreboard. Check your connection, "
            "nba_api version and CDN availability."
        ) from e

    board = raw.get("scoreboard", {})
    board_date = as_date(board.get("gameDate", "1900-01-01"))
    if board_date != target:
        raise ValueError(
            f"NBA live scoreboard is for {board_date}, not {target}. "
            "This feed serves TODAY only in NBA time, not arbitrary dates. "
            "Try again later when NBA's scoreboard rolls over."
        )

    records = []
    for game in board.get("games", []):
        if int(game.get("gameStatus", 0)) != 1:
            print(f"  SKIP {game.get('gameId')}: already live/final/not scheduled", flush=True)
            continue
        home = game["homeTeam"]
        away = game["awayTeam"]
        utc_time = datetime.fromisoformat(game["gameTimeUTC"].replace("Z", "+00:00"))
        local_time = utc_time.astimezone(ET)
        if local_time.date() != target:
            print(f"  SKIP {game.get('gameId')}: tipoff on {local_time.date()} ET", flush=True)
            continue
        records.append({
            "GAME_ID": str(game["gameId"]).zfill(10),
            "GAME_DATE": pd.Timestamp(target),
            "TIPOFF_ET": local_time.strftime("%Y-%m-%d %I:%M %p %Z"),
            "TIPOFF_UTC": utc_time.isoformat(),
            "HOME_TEAM": str(home["teamTricode"]),
            "AWAY_TEAM": str(away["teamTricode"]),
        })
    out = pd.DataFrame(records)
    if not out.empty and out.GAME_ID.duplicated().any():
        raise ValueError("Duplicate NBA scoreboard GAME_ID values")
    return out


def fetch_espn_scoreboard(target: date) -> list:
    import requests

    url = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
    response = requests.get(url, params={"dates": target.strftime("%Y%m%d"), "limit": 100},
                            headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"},
                            timeout=30)
    response.raise_for_status()
    events = response.json().get("events")
    if not isinstance(events, list):
        raise ValueError(f"ESPN scoreboard lacks events for {target}")
    return events


def fetch_espn_schedule(target: date, include_completed: bool = False) -> pd.DataFrame:
    """Fallback date-specific NBA schedule via ESPN's public scoreboard JSON."""
    records = []
    for event in fetch_espn_scoreboard(target):
        competitions = event.get("competitions", [])
        if len(competitions) != 1:
            continue
        competition = competitions[0]
        state = competition.get("status", {}).get("type", {}).get("state", "")
        if not include_completed and state != "pre":
            continue
        teams = {}
        for entry in competition.get("competitors", []):
            side = entry.get("homeAway")
            team = entry.get("team", {})
            if side in ("home", "away"):
                teams[side] = normalize_schedule_team(team.get("abbreviation", ""))
        if not teams.get("home") or not teams.get("away"):
            continue
        utc_time = datetime.fromisoformat(event["date"].replace("Z", "+00:00"))
        local_time = utc_time.astimezone(ET)
        if local_time.date() != target:
            continue
        records.append({
            "GAME_ID": "ESPN-" + str(event["id"]),
            "GAME_DATE": pd.Timestamp(target),
            "TIPOFF_ET": local_time.strftime("%Y-%m-%d %I:%M %p %Z"),
            "TIPOFF_UTC": utc_time.isoformat(),
            "HOME_TEAM": teams["home"], "AWAY_TEAM": teams["away"],
        })
    result = pd.DataFrame(records, columns=["GAME_ID", "GAME_DATE", "TIPOFF_ET",
                                            "TIPOFF_UTC", "HOME_TEAM", "AWAY_TEAM"])
    if result.GAME_ID.duplicated().any():
        raise ValueError("Duplicate ESPN schedule game IDs")
    return result


def fetch_dated_schedule(target: date, include_completed: bool = False) -> pd.DataFrame:
    """Date-specific NBA full-season schedule, independent of live scoreboard rollover."""
    import requests

    url = "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2_1.json"
    try:
        response = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        print(f"NBA CDN schedule unavailable ({exc}); trying ESPN date-specific scoreboard...", flush=True)
        try:
            return fetch_espn_schedule(target, include_completed=include_completed)
        except Exception as fallback_exc:
            raise RuntimeError(
                f"Both NBA CDN and ESPN schedule requests failed. "
                f"NBA: {exc}; ESPN: {fallback_exc}"
            ) from fallback_exc

    groups = payload.get("leagueSchedule", {}).get("gameDates", [])
    records = []
    for group in groups:
        for game in group.get("games", []):
            if not include_completed and int(game.get("gameStatus", 0)) != 1:
                continue
            time_str = game.get("gameDateTimeUTC") or game.get("gameTimeUTC")
            if not time_str:
                continue
            utc_time = datetime.fromisoformat(time_str.replace("Z", "+00:00"))
            if utc_time.tzinfo is None:
                utc_time = utc_time.replace(tzinfo=timezone.utc)
            local_time = utc_time.astimezone(ET)
            if local_time.date() != target:
                continue
            home, away = game.get("homeTeam", {}), game.get("awayTeam", {})
            if not home.get("teamTricode") or not away.get("teamTricode"):
                continue
            records.append({
                "GAME_ID": str(game["gameId"]).zfill(10),
                "GAME_DATE": pd.Timestamp(target),
                "TIPOFF_ET": local_time.strftime("%Y-%m-%d %I:%M %p %Z"),
                "TIPOFF_UTC": utc_time.isoformat(),
                "HOME_TEAM": str(home["teamTricode"]),
                "AWAY_TEAM": str(away["teamTricode"]),
            })
    result = pd.DataFrame(records)
    if not result.empty and result.GAME_ID.duplicated().any():
        raise ValueError("Duplicate NBA schedule GAME_ID values")
    return result


def load_history(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run the original cleaning pipeline first.")
    old = pd.read_csv(path, dtype={"GAME_ID": str, "TEAM_ID": str})
    required = {"GAME_ID", "GAME_DATE", "SEASON", "TEAM_ID", "TEAM_ABBREVIATION", "WL", "PTS", "IS_HOME"}
    missing = required - set(old.columns)
    if missing:
        raise ValueError(f"Historical CSV missing columns: {sorted(missing)}")
    return old[list(required)].copy()


def fetch_current_season(season: str, csv_path: Path | None) -> pd.DataFrame:
    if csv_path is not None:
        if not csv_path.exists():
            raise FileNotFoundError(csv_path)
        print(f"Reading current season from {csv_path}", flush=True)
        df = pd.read_csv(csv_path, dtype={"GAME_ID": str, "TEAM_ID": str})
    else:
        try:
            from nba_api.stats.endpoints import leaguegamefinder
            print(f"Downloading completed {season} regular-season team-game results...", flush=True)
            df = leaguegamefinder.LeagueGameFinder(
                player_or_team_abbreviation="T",
                league_id_nullable="00",
                season_nullable=season,
                season_type_nullable="Regular Season",
                timeout=35,
            ).get_data_frames()[0]
            if df.empty:
                raise ValueError("NBA Stats returned no completed games")
        except Exception:
            from .espn_season import fetch_espn_season

            print(f"NBA Stats unavailable for {season}; using ESPN final scores.", flush=True)
            df = fetch_espn_season(season, datetime.now(ET).date(), DATA / "espn_nba_results",
                                   fetch_espn_scoreboard, normalize_schedule_team)
        # Preserve source data, never overwrite original 4-season files.
        cache = DATA / f"nba_{season.replace('-', '_')}_latest_raw.csv"
        df.to_csv(cache, index=False)
        print(f"Saved fresh season data to {cache}", flush=True)

    required = {"GAME_ID", "GAME_DATE", "TEAM_ID", "TEAM_ABBREVIATION", "WL", "PTS"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Current-season data missing columns {sorted(missing)}")
    if "IS_HOME" not in df.columns:
        if "MATCHUP" not in df.columns:
            raise ValueError("Season CSV requires either IS_HOME or MATCHUP (NBA game log format)")
            # Determine home/away from matchup and team abbreviation.
    # Handles exports where both rows use "AWAY @ HOME".
    matchups = df["MATCHUP"].astype(str).str.strip()
    teams = df["TEAM_ABBREVIATION"].astype(str).str.strip().str.upper()

    is_at = matchups.str.contains(" @ ", regex=False)
    is_vs = matchups.str.contains(" vs. ", regex=False)

    # In "DEN @ IND", IND is the home team.
    home_from_at = matchups.str.split(" @ ").str[-1].str.strip().str.upper()

    # In "IND vs. DEN", IND is the home team.
    home_from_vs = matchups.str.split(" vs. ").str[0].str.strip().str.upper()

    df["IS_HOME"] = (
        (is_at & teams.eq(home_from_at))
        | (is_vs & teams.eq(home_from_vs))
    )
        # df["IS_HOME"] = df["MATCHUP"].astype(str).str.contains("vs.", regex=False)
    df["SEASON"] = season
    return df[["GAME_ID", "GAME_DATE", "SEASON", "TEAM_ID", "TEAM_ABBREVIATION", "WL", "PTS", "IS_HOME"]].copy()


def clean_team_history(old: pd.DataFrame, current: pd.DataFrame, target: date) -> pd.DataFrame:
    def normalize(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["GAME_ID"] = df["GAME_ID"].astype(str).str.zfill(10)
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"], errors="raise").dt.normalize()
        df["TEAM_ID"] = df["TEAM_ID"].astype(str)
        df["TEAM_ABBREVIATION"] = df["TEAM_ABBREVIATION"].astype(str).str.upper().str.strip()
        df["PTS"] = pd.to_numeric(df["PTS"], errors="coerce")
        # pandas can read IS_HOME as bool, or strings in externally prepared exports.
        if df["IS_HOME"].dtype == bool:
            pass
        else:
            values = df["IS_HOME"].astype(str).str.lower().str.strip()
            if not values.isin(["true", "false", "1", "0"]).all():
                raise ValueError("IS_HOME requires true/false values")
            df["IS_HOME"] = values.isin(["true", "1"])
        return df

    all_games = pd.concat([normalize(old), normalize(current)], ignore_index=True)
    all_games = all_games.loc[all_games["GAME_DATE"] < pd.Timestamp(target)].copy()
    if all_games.empty:
        raise ValueError("No historical games before forecast date")
    if all_games.duplicated(["GAME_ID", "TEAM_ID"]).any():
        raise ValueError("Duplicate game/team rows across historical and current season sources")
    if all_games[["PTS", "WL"]].isna().any().any() or not all_games.WL.isin(["W", "L"]).all():
        raise ValueError("Season data contains unfinished games or missing scores before target date")
    checks = all_games.groupby("GAME_ID").agg(
        rows=("TEAM_ID", "size"), homes=("IS_HOME", "sum"),
        wins=("WL", lambda s: int((s == "W").sum())),
        dates=("GAME_DATE", "nunique"), seasons=("SEASON", "nunique"),
    )
    bad = checks[(checks.rows != 2) | (checks.homes != 1) | (checks.wins != 1) |
                 (checks.dates != 1) | (checks.seasons != 1)]
    if not bad.empty:
        raise ValueError(
            f"{len(bad)} game(s) have incomplete/inconsistent two-team logs. "
            f"First invalid GAME_IDs: {list(bad.index[:5])}. "
            "Verify the current-season export is COMPLETE and uses regular-season games only."
        )
    return all_games.sort_values(["GAME_DATE", "GAME_ID", "TEAM_ID"]).reset_index(drop=True)


def make_completed_games(team_games: pd.DataFrame) -> pd.DataFrame:
    home = team_games.loc[team_games.IS_HOME, [
        "GAME_ID", "GAME_DATE", "SEASON", "TEAM_ABBREVIATION", "WL", "PTS"
    ]].rename(columns={"TEAM_ABBREVIATION": "HOME_TEAM", "PTS": "HOME_PTS"})
    away = team_games.loc[~team_games.IS_HOME, [
        "GAME_ID", "TEAM_ABBREVIATION", "PTS"
    ]].rename(columns={"TEAM_ABBREVIATION": "AWAY_TEAM", "PTS": "AWAY_PTS"})
    games = home.merge(away, on="GAME_ID", validate="one_to_one")
    games["HOME_WIN"] = games["WL"].eq("W").astype(int)
    return games.sort_values(["GAME_DATE", "GAME_ID"]).reset_index(drop=True)


def compute_elo(completed: pd.DataFrame, season: str) -> dict[str, float]:
    """Replay src.elo EXACT rules: batch by date, home advantage, offseason regression."""
    ratings: dict[str, float] = {}
    previous_season = None
    for game_date, day in completed.groupby("GAME_DATE", sort=True):
        season_values = day.SEASON.astype(str).unique()
        if len(season_values) != 1:
            raise ValueError(f"Multiple seasons on {game_date}: {season_values}")
        this_season = str(season_values[0])
        if previous_season is not None and this_season != previous_season:
            ratings = {team: INITIAL_ELO + (r - INITIAL_ELO) * (1 - OFFSEASON_REGRESSION)
                       for team, r in ratings.items()}
        previous_season = this_season
        updates: dict[str, float] = {}
        for row in day.itertuples(index=False):
            h, a = row.HOME_TEAM, row.AWAY_TEAM
            rh, ra = ratings.get(h, INITIAL_ELO), ratings.get(a, INITIAL_ELO)
            expected = 1.0 / (1.0 + 10.0 ** (-(rh - ra + HOME_ADVANTAGE) / 400.0))
            delta = K_FACTOR * (int(row.HOME_WIN) - expected)
            updates[h] = updates.get(h, 0.0) + delta
            updates[a] = updates.get(a, 0.0) - delta
        for team, delta in updates.items():
            ratings[team] = ratings.get(team, INITIAL_ELO) + delta
    if previous_season != season:
        ratings = {team: INITIAL_ELO + (r - INITIAL_ELO) * (1 - OFFSEASON_REGRESSION)
                   for team, r in ratings.items()}
    return ratings


def snapshot_team_stats(team_games: pd.DataFrame, target: date) -> dict[str, dict]:
    """Equivalent to src.feature_engineering's shifted windows, at a future date."""
    teams = {}
    for abbrev, games in team_games.groupby("TEAM_ABBREVIATION"):
        games = games.sort_values(["GAME_DATE", "GAME_ID"])
        # A single row represents one team's completed game; grouping by TEAM_ID
        # in original feature generator is equivalent for NBA franchises.
        if len(games) < 10:
            continue
        last10 = games.tail(10)
        win_flags = games["WL"].eq("W").astype(float)
        last_date = games["GAME_DATE"].iloc[-1].date()
        teams[abbrev] = {
            "WIN_PCT_LAST_5": float(win_flags.tail(5).mean()),
            "WIN_PCT_LAST_10": float(win_flags.tail(10).mean()),
            "DAYS_REST": max(0, (target - last_date).days - 1),
            "POINT_DIFF_LAST_10": float(last10["POINT_DIFF"].mean()),
            "HISTORY_GAMES": int(len(games)),
            "LAST_GAME": str(last_date),
        }
    return teams


def build_forecasts(schedule: pd.DataFrame, team_games: pd.DataFrame, model_bundle: dict,
                    season: str, target: date) -> pd.DataFrame:
    completed = make_completed_games(team_games)
    elos = compute_elo(completed, season)
    team_games = team_games.copy()
    # Exactly two team rows per completed game (validated above).
    team_games["POINT_DIFF"] = (2 * team_games["PTS"] -
                                team_games.groupby("GAME_ID")["PTS"].transform("sum"))
    stats = snapshot_team_stats(team_games, target)
    inputs = model_bundle["inputs"]
    if inputs != EXPECTED_INPUTS:
        raise ValueError(
            f"Saved model inputs differ from locked selected Elo model. "
            f"Expected {EXPECTED_INPUTS}, got {inputs}."
        )

    forecasts = []
    for game in schedule.itertuples(index=False):
        h, a = game.HOME_TEAM, game.AWAY_TEAM
        if h not in stats or a not in stats:
            raise ValueError(f"Insufficient recent team games for {a} @ {h} — "
                             "check the 2026-27 season download")
        values = {}
        for side, abbr in (("HOME", h), ("AWAY", a)):
            for name in ("WIN_PCT_LAST_5", "WIN_PCT_LAST_10", "DAYS_REST", "POINT_DIFF_LAST_10"):
                values[f"{side}_{name}"] = stats[abbr][name]
        home_elo = elos.get(h, INITIAL_ELO)
        away_elo = elos.get(a, INITIAL_ELO)
        values["ELO_DIFF"] = home_elo - away_elo
        sample = pd.DataFrame([values], columns=inputs)
        p_gb = float(model_bundle["model"].predict_proba(sample)[0, 1])
        p_elo = float(1.0 / (1.0 + 10.0 ** (-(values["ELO_DIFF"] + HOME_ADVANTAGE) / 400.0)))
        forecasts.append({
            "GAME_DATE": str(target), "GAME_ID": game.GAME_ID,
            "TIPOFF_ET": game.TIPOFF_ET, "TIPOFF_UTC": game.TIPOFF_UTC,
            "AWAY_TEAM": a, "HOME_TEAM": h,
            "P_HOME_WIN_GB": p_gb, "P_AWAY_WIN_GB": 1-p_gb,
            "P_HOME_WIN_ELO": p_elo, "P_AWAY_WIN_ELO": 1-p_elo,
            "GB_PICK": h if p_gb >= 0.5 else a,
            "ELO_PICK": h if p_elo >= 0.5 else a,
            "HOME_ELO": home_elo, "AWAY_ELO": away_elo,
            "ELO_DIFF": values["ELO_DIFF"],
            "HOME_LAST_GAME": stats[h]["LAST_GAME"],
            "AWAY_LAST_GAME": stats[a]["LAST_GAME"],
            **values,
        })
    return pd.DataFrame(forecasts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=str(datetime.now(ET).date()),
                        help="NBA Eastern calendar date, YYYY-MM-DD (default: today ET)")
    parser.add_argument("--season-csv", type=Path, default=None,
                        help="Optional full current-season NBA LeagueGameFinder CSV")
    parser.add_argument("--model", choices=["v1", "v2"], default="v1", help="Choose frozen baseline or current-season retrained model")
    args = parser.parse_args()
    target = as_date(args.date)
    season = f"{target.year}-{(target.year + 1) % 100:02d}" if target.month >= 7 else \
             f"{target.year - 1}-{target.year % 100:02d}"
    print(f"===== PARLAY GUARD PRE-TIPOFF FORECAST: {target} ({season}) =====", flush=True)
    model_path = MODELS / ("elo_2026_27_v2.joblib" if args.model == "v2" else "elo_final_holdout_model.joblib")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing {model_path}; first run python -u -m src.final_validation")
    model_bundle = joblib.load(model_path)  # Only load joblib files you generated and trust.
    if not isinstance(model_bundle, dict) or not {"model", "inputs"}.issubset(model_bundle):
        raise ValueError("Unexpected model bundle; need final_validation.py's saved model")
    print(f"Loaded LOCKED model from {model_path} (no refit)", flush=True)

    print("Fetching date-specific NBA schedule (no scoreboard rollover required)...", flush=True)
    schedule = fetch_dated_schedule(target)
    if schedule.empty:
        print("No unstarted NBA games remain for this date. No predictions made.", flush=True)
        return
    print(f"Unstarted games today: {len(schedule)}", flush=True)

    historical = load_history(DATA / "nba_games_updated.csv")
    current = fetch_current_season(season, args.season_csv)
    # Guard against a partial current-season download: if a team in today's
    # schedule lacks sufficient current-season history, do not silently predict.
    today_teams = set(schedule.HOME_TEAM) | set(schedule.AWAY_TEAM)
    season_dates = pd.to_datetime(current.GAME_DATE, errors="raise")
    recent = current.loc[season_dates < pd.Timestamp(target)]
    if recent.empty:
        raise ValueError("No completed current-season games available before prediction date")
    latest_current_date = pd.to_datetime(recent.GAME_DATE).max().date()
    if (target - latest_current_date).days > 3:
        raise ValueError(
            f"Current-season game log appears stale (latest completed game "
            f"{latest_current_date}, forecast date {target}). "
            "Refusing to use outdated rolling statistics or Elo ratings."
        )
    print(f"Latest current-season completed game: {latest_current_date}", flush=True)
    current_counts = recent.groupby("TEAM_ABBREVIATION").GAME_ID.nunique()
    missing_season_teams = sorted(team for team in today_teams if current_counts.get(team, 0) < 5)
    if missing_season_teams:
        raise ValueError(
            f"Current season has fewer than 5 completed games for {missing_season_teams}. "
            "Season download may be incomplete or missing. Refusing stale predictions."
        )
    
    history = clean_team_history(historical, current, target)
    print(f"Completed games available through {target}: {history.GAME_ID.nunique():,}", flush=True)
    forecast = build_forecasts(schedule, history, model_bundle, season, target)
    now = datetime.now(timezone.utc)
    # Never present a post-tipoff generated prediction as pregame, even if
    # scoreboard happens to lag the actual scheduled start time.
    before_tipoff = pd.to_datetime(forecast["TIPOFF_UTC"], utc=True) > pd.Timestamp(now)
    if not before_tipoff.all():
        excluded = forecast.loc[~before_tipoff, ["AWAY_TEAM", "HOME_TEAM", "TIPOFF_ET"]]
        for row in excluded.itertuples(index=False):
            print(f"  SKIP {row.AWAY_TEAM} @ {row.HOME_TEAM}: already past scheduled tipoff {row.TIPOFF_ET}",
                  flush=True)
        forecast = forecast.loc[before_tipoff].copy()
    if forecast.empty:
        print("No games remain before tipoff; no pregame predictions saved.", flush=True)
        return
    forecast.insert(0, "GENERATED_AT_UTC", now.isoformat())
    output = MODELS / f"live_predictions_{args.model}_{target}_{now.strftime('%Y%m%dT%H%M%SZ')}.csv"
    forecast.sort_values("TIPOFF_UTC").to_csv(output, index=False)

    cols = ["TIPOFF_ET", "AWAY_TEAM", "HOME_TEAM", "P_HOME_WIN_GB", "P_HOME_WIN_ELO", "GB_PICK", "ELO_PICK"]
    print("\n===== UNSEEN FUTURE NBA GAMES — PREDICTIONS =====", flush=True)
    print(forecast[cols].to_string(index=False, formatters={
        "P_HOME_WIN_GB": lambda x: f"{x:.1%}",
        "P_HOME_WIN_ELO": lambda x: f"{x:.1%}",
    }), flush=True)
    print(f"\nSaved immutable pre-tipoff forecast: {output}", flush=True)
    print("IMPORTANT: Both probabilities are HOME win. Away win = 1 minus home win.", flush=True)
    print("Freeze this CSV; after games finish, compare to results, don't edit predictions.", flush=True)
    print(f"Model version: {args.model}; model training cutoff: {model_bundle.get('training_cutoff_exclusive', model_bundle.get('last_historical_date', 'unknown'))}", flush=True)
    print("Verify injuries/starting lineups separately; they are not included.", flush=True)


if __name__ == "__main__":
    main()
