
"""Build leakage-safe NBA player-stat training features."""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

INPUT = ROOT / "data" / "nba_player_stats_history.csv"
OUTPUT = ROOT / "data" / "processed" / "player_training_data_v2.csv"

TARGETS = ["PTS", "REB", "AST", "FG3M", "STL", "BLK"]

HISTORY_STATS = TARGETS + [
    "MIN", "FGA", "FG3A", "FTA", "TOV"
]

WINDOWS = [5, 10, 20]


def build_features(df):
    df = df.copy()

    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

    df = df.sort_values(
        ["PLAYER_ID", "GAME_DATE", "GAME_ID"]
    ).reset_index(drop=True)

    df["IS_HOME"] = (
        df["MATCHUP"].str.contains("vs.", regex=False)
        .astype(int)
    )

    player_groups = df.groupby("PLAYER_ID", sort=False)

    previous_date = player_groups["GAME_DATE"].shift(1)

    df["REST_DAYS"] = (
        df["GAME_DATE"] - previous_date
    ).dt.days - 1

    df["REST_DAYS"] = df["REST_DAYS"].clip(
        lower=0, upper=14
    )

    df["GAMES_PLAYED_BEFORE"] = player_groups.cumcount()

    # Count previous appearances in the current season.
    season_groups = df.groupby(
        ["PLAYER_ID", "SEASON"], sort=False
    )

    df["CURRENT_SEASON_GAMES_PLAYED"] = (
        season_groups.cumcount()
    )

    feature_columns = [
        "IS_HOME",
        "REST_DAYS",
        "GAMES_PLAYED_BEFORE",
        "CURRENT_SEASON_GAMES_PLAYED",
    ]

    # Historical averages and volatility.
    for stat in HISTORY_STATS:
        df[stat] = pd.to_numeric(
            df[stat], errors="coerce"
        )

        groups = df.groupby("PLAYER_ID", sort=False)[stat]

        for window in WINDOWS:
            avg_name = f"{stat}_LAST_{window}"
            std_name = f"{stat}_STD_{window}"

            df[avg_name] = groups.transform(
                lambda s: s.shift(1).rolling(
                    window=window,
                    min_periods=window
                ).mean()
            )

            df[std_name] = groups.transform(
                lambda s: s.shift(1).rolling(
                    window=window,
                    min_periods=window
                ).std()
            )

            feature_columns.extend([
                avg_name, std_name
            ])

    # Explicit changes in player role and production.
    for stat in ["PTS", "MIN", "FGA", "FG3A", "AST"]:
        trend_name = f"{stat}_TREND_5_20"

        df[trend_name] = (
            df[f"{stat}_LAST_5"]
            - df[f"{stat}_LAST_20"]
        )

        feature_columns.append(trend_name)

    # Current-season scoring and playing-time averages.
    for stat in ["PTS", "MIN", "FGA"]:
        name = f"{stat}_SEASON_AVG"

        df[name] = df.groupby(
            ["PLAYER_ID", "SEASON"]
        )[stat].transform(
            lambda s: s.shift(1).expanding(
                min_periods=1
            ).mean()
        )

        # Fall back to long-term history for season openers.
        df[name] = df[name].fillna(
            df[f"{stat}_LAST_20"]
        )

        feature_columns.append(name)

    training = df[
        df["GAMES_PLAYED_BEFORE"] >= 20
    ].copy()

    training = training.dropna(
        subset=feature_columns + TARGETS
    )

    return training, feature_columns


def main():
    df = pd.read_csv(
        INPUT,
        dtype={
            "GAME_ID": str,
            "PLAYER_ID": str
        }
    )

    training, features = build_features(df)

    OUTPUT.parent.mkdir(
        parents=True, exist_ok=True
    )

    training.to_csv(OUTPUT, index=False)

    print("Training rows:", len(training))
    print("Features:", len(features))
    print("Players:", training["PLAYER_ID"].nunique())
    print("Missing features:",
          training[features].isna().sum().sum())
    print("Saved to:", OUTPUT)


if __name__ == "__main__":
    main()
