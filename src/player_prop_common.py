"""Shared feature definitions for Parlay Guard player-prop models.

Keep FEATURES in this file as the single source of truth for training/inference.
"""
from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

TARGETS = ["PTS", "REB", "AST", "FG3M"]
ROLLING_STATS = ["PTS", "REB", "AST", "FG3M", "MIN", "FGA", "FG3A", "FTA", "TOV"]
FEATURES = ["IS_HOME", "REST_DAYS", "GAMES_PLAYED_BEFORE", "CURRENT_SEASON_GAMES_PLAYED"]
for _stat in ROLLING_STATS:
    for _window in (5, 10, 20):
        FEATURES.extend([f"{_stat}_LAST_{_window}", f"{_stat}_STD_{_window}"])
FEATURES += [
    "PTS_TREND_5_20", "MIN_TREND_5_20", "FGA_TREND_5_20",
    "FG3A_TREND_5_20", "AST_TREND_5_20",
    "PTS_SEASON_AVG", "MIN_SEASON_AVG", "FGA_SEASON_AVG",
]

# Official NBA team identifiers, for matching a player's most recently observed
# team with an event slug such as nba-gsw-mem-2026-11-19.
TEAM_ID_TO_ABBR = {
    1610612737: "atl", 1610612738: "bos", 1610612751: "bkn",
    1610612766: "cha", 1610612741: "chi", 1610612739: "cle",
    1610612742: "dal", 1610612743: "den", 1610612765: "det",
    1610612744: "gsw", 1610612745: "hou", 1610612754: "ind",
    1610612746: "lac", 1610612747: "lal", 1610612763: "mem",
    1610612748: "mia", 1610612749: "mil", 1610612750: "min",
    1610612740: "nop", 1610612752: "nyk", 1610612760: "okc",
    1610612753: "orl", 1610612755: "phi", 1610612756: "phx",
    1610612757: "por", 1610612758: "sac", 1610612759: "sas",
    1610612761: "tor", 1610612762: "uta", 1610612764: "was",
}


def name_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).casefold())


def season_of_date(day: pd.Timestamp) -> str:
    year = day.year if day.month >= 7 else day.year - 1
    return f"{year}-{(year + 1) % 100:02d}"


def prepare_history(history: pd.DataFrame) -> pd.DataFrame:
    needed = {"GAME_DATE", "PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "SEASON"} | set(ROLLING_STATS)
    missing = needed - set(history.columns)
    if missing:
        raise ValueError(f"Historical CSV missing columns: {sorted(missing)}")
    history = history.copy()
    history["GAME_DATE"] = pd.to_datetime(history["GAME_DATE"], format="mixed", errors="coerce").dt.normalize()
    history = history.dropna(subset=["GAME_DATE", "PLAYER_ID", "PLAYER_NAME"])
    for stat in ROLLING_STATS:
        history[stat] = pd.to_numeric(history[stat], errors="coerce")
    # Forecast only from actually completed appearances, not future schedule rows.
    history = history.dropna(subset=["PTS", "REB", "AST", "FG3M", "MIN"])
    if "IS_HOME" in history.columns:
        home = history["IS_HOME"].replace({"True": 1, "False": 0, "true": 1, "false": 0, True: 1, False: 0})
        history["IS_HOME"] = pd.to_numeric(home, errors="coerce")
    history["PLAYER_KEY"] = history["PLAYER_NAME"].map(name_key)
    # Prevent repeated player-game rows from inflating rolling statistics.
    if "GAME_ID" in history.columns:
        history = history.drop_duplicates(["PLAYER_ID", "GAME_ID"], keep="last")
    history = history.sort_values(["PLAYER_ID", "GAME_DATE", "GAME_ID"] if "GAME_ID" in history.columns else ["PLAYER_ID", "GAME_DATE"])
    return history.reset_index(drop=True)


def build_feature_row(previous: pd.DataFrame, *, target_day: pd.Timestamp,
                      season: str, is_home: int, rest_offset: int = -1) -> dict:
    """Reconstruct next-game features from the *retained* engineered training rows.

    Important: player_training_data_v2.csv is NOT a complete raw game log.
    It drops a player's earliest games when 20-game rolling features aren't
    available.  Counting retained rows therefore undercounts career games.

    Counter and season-average values are advanced from the LAST STORED
    pregame values, which preserve information about the trimmed appearances.
    Rolling windows still require 20 genuinely retained, completed appearances.
    """
    if previous.empty:
        raise ValueError("No previous completed appearances")
    order = ["GAME_DATE", "GAME_ID"] if "GAME_ID" in previous.columns else ["GAME_DATE"]
    previous = previous.sort_values(order)
    if not (previous["GAME_DATE"] < target_day).all():
        raise ValueError("Leakage guard: game on/after target date")
    if len(previous) < 20:
        raise ValueError("Fewer than 20 retained appearances; cannot reconstruct 20-game features")
    last = previous.iloc[-1]
    last_day = last["GAME_DATE"]
    count_columns = ["GAMES_PLAYED_BEFORE", "CURRENT_SEASON_GAMES_PLAYED"]
    for key in count_columns:
        if key not in previous or pd.isna(last[key]):
            raise ValueError(f"Missing pregame counter anchor: {key}")

    # A historical row records appearances *before* the game on that row.
    # Advance by one for that completed game; never count trimmed CSV rows.
    all_games_before = int(last["GAMES_PLAYED_BEFORE"]) + 1
    same_season = str(last["SEASON"]) == season
    season_games_before = (int(last["CURRENT_SEASON_GAMES_PLAYED"]) + 1
                           if same_season else 0)
    row = {
        "IS_HOME": int(is_home),
        "REST_DAYS": max(0, (target_day - last_day).days + rest_offset),
        "GAMES_PLAYED_BEFORE": all_games_before,
        "CURRENT_SEASON_GAMES_PLAYED": season_games_before,
    }
    for stat in ROLLING_STATS:
        values = pd.to_numeric(previous[stat], errors="coerce")
        for window in (5, 10, 20):
            tail = values.tail(window)
            # The original features use a *full* window. Reject a partially
            # missing window instead of silently treating it as a valid mean.
            row[f"{stat}_LAST_{window}"] = tail.mean() if tail.notna().all() else np.nan
            row[f"{stat}_STD_{window}"] = tail.std(ddof=1) if tail.notna().all() else np.nan
    for stat in ("PTS", "MIN", "FGA", "FG3A", "AST"):
        row[f"{stat}_TREND_5_20"] = row[f"{stat}_LAST_5"] - row[f"{stat}_LAST_20"]

    for stat in ("PTS", "MIN", "FGA"):
        # Historical season averages were computed as shifted expanding means,
        # with LAST_20 as fallback for the season's first game.  The final
        # stored pregame average encodes earlier appearances absent from the
        # feature-selected CSV. Update it with the final observed performance.
        if not same_season:
            row[f"{stat}_SEASON_AVG"] = row[f"{stat}_LAST_20"]
        elif season_games_before == 1:
            row[f"{stat}_SEASON_AVG"] = float(last[stat])
        else:
            previous_avg = pd.to_numeric(pd.Series([last[f"{stat}_SEASON_AVG"]]), errors="coerce").iloc[0]
            observed_stat = pd.to_numeric(pd.Series([last[stat]]), errors="coerce").iloc[0]
            if pd.isna(previous_avg) or pd.isna(observed_stat):
                row[f"{stat}_SEASON_AVG"] = np.nan
            else:
                prior_n = season_games_before - 1
                row[f"{stat}_SEASON_AVG"] = ((float(previous_avg) * prior_n)
                                               + float(observed_stat)) / season_games_before
    return row


def audit_feature_parity(history: pd.DataFrame, *, samples: int = 32) -> tuple[bool, list[str], int]:
    """Check *all* trained features against real historical pregame values.

    This also detects if early raw games were trimmed, or if the original
    training features use different rolling, rest, or season-mean formulas.
    """
    missing = sorted(set(FEATURES) - set(history.columns))
    if missing:
        return False, [f"Missing training features for audit: {missing}"], 0
    if "IS_HOME" not in history:
        return False, ["No historical IS_HOME values"], 0
    past_count = history.groupby("PLAYER_ID").cumcount()
    candidates = history[(past_count >= 20) & history["IS_HOME"].notna()]
    if candidates.empty:
        return False, ["No eligible rows with 20 retained prior appearances"], 0
    candidates = candidates.sort_values(["GAME_DATE", "PLAYER_ID"]).tail(samples)

    from collections import Counter
    counts = Counter()
    examples: dict[str, str] = {}
    checked = 0
    for _, observed in candidates.iterrows():
        player_history = history[(history["PLAYER_ID"] == observed["PLAYER_ID"])
                                 & (history["GAME_DATE"] < observed["GAME_DATE"])]
        if len(player_history) < 20:
            continue
        home = pd.to_numeric(pd.Series([observed["IS_HOME"]]), errors="coerce").iloc[0]
        if pd.isna(home):
            continue
        try:
            expected = build_feature_row(player_history, target_day=observed["GAME_DATE"],
                                         season=str(observed["SEASON"]), is_home=int(home))
        except ValueError as exc:
            return False, [str(exc)], checked
        checked += 1
        for col in FEATURES:
            actual = pd.to_numeric(pd.Series([observed[col]]), errors="coerce").iloc[0]
            derived = expected[col]
            if pd.isna(actual) and pd.isna(derived):
                continue
            mismatch = (pd.isna(actual) or pd.isna(derived) or
                        not np.isclose(float(actual), float(derived), atol=0.011, rtol=0.001))
            if mismatch:
                counts[col] += 1
                if col not in examples:
                    examples[col] = (f"{col}: {observed['PLAYER_NAME']} on "
                                     f"{observed['GAME_DATE'].date()} "
                                     f"historical={actual} reconstructed={derived}")
    issues: list[str] = []
    if counts:
        issues.append(f"Mismatching feature columns: {len(counts)}/{len(FEATURES)}")
        for col, n in counts.most_common(12):
            issues.append(f"{n}/{checked} rows | {examples[col]}")
    if checked < 5:
        issues.append(f"Only {checked} comparable rows; need at least five")
    return checked >= 5 and not issues, issues, checked
