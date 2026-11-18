
"""Predict NBA player stats using V1 or V2 CatBoost models.

Examples:
    python -m src.player_models.predict --player "Luka Dončić" --opponent BOS --home --date 2026-11-19

    python -m src.player_models.predict --player "Luka Dončić" --opponent BOS --home --date 2026-11-19 --model-version v1

    python -m src.player_models.predict --player "Stephen Curry" --opponent LAL --away --date 2026-11-19 --json
"""

import argparse
import json
import unicodedata
from pathlib import Path

import pandas as pd
from catboost import CatBoostRegressor


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "nba_player_stats_history.csv"

MODEL_DIRS = {
    "v1": ROOT / "models" / "player_stats",
    "v2": ROOT / "models" / "player_stats_v2",
}

TARGETS = ["PTS", "REB", "AST", "FG3M", "STL", "BLK"]

HISTORY_STATS = TARGETS + [
    "MIN", "FGA", "FG3A", "FTA", "TOV"
]

WINDOWS = [5, 10, 20]


def normalize_name(name):
    text = unicodedata.normalize("NFKD", str(name))
    text = "".join(
        char for char in text
        if not unicodedata.combining(char)
    )
    return text.casefold().strip()


def load_history():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Historical data not found: {DATA_PATH}"
        )

    df = pd.read_csv(
        DATA_PATH,
        dtype={"GAME_ID": str, "PLAYER_ID": str},
    )

    df["GAME_DATE"] = pd.to_datetime(
        df["GAME_DATE"], errors="raise"
    )

    required = [
        "PLAYER_ID", "PLAYER_NAME", "GAME_ID",
        "GAME_DATE", "SEASON", *HISTORY_STATS
    ]

    missing = [
        column for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Historical data missing columns: {missing}"
        )

    return df


def find_player(df, player_name):
    target = normalize_name(player_name)

    names = df["PLAYER_NAME"].dropna().unique()

    exact = [
        name for name in names
        if normalize_name(name) == target
    ]

    if len(exact) == 1:
        return exact[0]

    partial = [
        name for name in names
        if target in normalize_name(name)
    ]

    if len(partial) == 1:
        return partial[0]

    if len(partial) > 1:
        raise ValueError(
            "Multiple players matched: "
            + ", ".join(sorted(partial))
        )

    raise ValueError(
        f"Player '{player_name}' not found."
    )


def season_for_date(game_date):
    """NBA regular-season label for an October-June date.

    This is a calendar-based convention, not a schedule lookup.
    Predictions outside that range require an explicit season.
    """
    year = game_date.year

    if game_date.month >= 10:
        start_year = year
    elif game_date.month <= 6:
        start_year = year - 1
    else:
        raise ValueError(
            "For July-September predictions, supply --season "
            "explicitly (e.g. 2026-27)."
        )

    return f"{start_year}-{str(start_year + 1)[-2:]}"


def build_prediction_features(
    df,
    player_name,
    game_date,
    is_home,
    season=None,
):
    player = find_player(df, player_name)

    player_rows = df[
        df["PLAYER_NAME"] == player
    ].copy()

    player_ids = player_rows["PLAYER_ID"].unique()

    if len(player_ids) != 1:
        raise ValueError(
            f"Ambiguous player ID for {player}: "
            f"{list(player_ids)}"
        )

    player_id = player_ids[0]

    history = df[
        (df["PLAYER_ID"] == player_id)
        & (df["GAME_DATE"] < game_date)
    ].copy()

    history = history.sort_values(
        ["GAME_DATE", "GAME_ID"]
    )

    if len(history) < 20:
        raise ValueError(
            f"{player} has only {len(history)} prior games; "
            "at least 20 are required."
        )

    current_season = season or season_for_date(game_date)

    # Exclude any historical records from a future season.
    season_history = history[
        history["SEASON"] == current_season
    ].copy()

    last_game_date = history["GAME_DATE"].iloc[-1]

    rest_days = (
        game_date - last_game_date
    ).days - 1

    if rest_days < 0:
        raise ValueError(
            "Game date must follow the last historical game."
        )

    features = {
        "IS_HOME": int(is_home),
        "REST_DAYS": min(rest_days, 14),
        "GAMES_PLAYED_BEFORE": len(history),
        "CURRENT_SEASON_GAMES_PLAYED": len(season_history),
    }

    for stat in HISTORY_STATS:
        values = pd.to_numeric(
            history[stat], errors="coerce"
        )

        if values.tail(20).isna().any():
            raise ValueError(
                f"Missing {stat} in recent player history."
            )

        for window in WINDOWS:
            recent = values.tail(window)

            features[f"{stat}_LAST_{window}"] = (
                float(recent.mean())
            )

            features[f"{stat}_STD_{window}"] = (
                float(recent.std(ddof=1))
            )

    # Same trend calculations used by features.py V2.
    for stat in ["PTS", "MIN", "FGA", "FG3A", "AST"]:
        features[f"{stat}_TREND_5_20"] = (
            features[f"{stat}_LAST_5"]
            - features[f"{stat}_LAST_20"]
        )

    # Match V2's expanding current-season averages.
    # At the season opener, V2 falls back to LAST_20.
    for stat in ["PTS", "MIN", "FGA"]:
        if len(season_history) > 0:
            season_values = pd.to_numeric(
                season_history[stat],
                errors="coerce"
            )

            if season_values.isna().any():
                raise ValueError(
                    f"Missing current-season {stat} values."
                )

            features[f"{stat}_SEASON_AVG"] = (
                float(season_values.mean())
            )
        else:
            features[f"{stat}_SEASON_AVG"] = (
                features[f"{stat}_LAST_20"]
            )

    feature_df = pd.DataFrame([features])

    recent_averages = {
        stat: {
            "last_5": round(
                features[f"{stat}_LAST_5"], 2
            ),
            "last_10": round(
                features[f"{stat}_LAST_10"], 2
            ),
            "last_20": round(
                features[f"{stat}_LAST_20"], 2
            ),
            "std_10": round(
                features[f"{stat}_STD_10"], 2
            ),
        }
        for stat in TARGETS
    }

    info = {
        "player": player,
        "player_id": str(player_id),
        "game_date": game_date.strftime("%Y-%m-%d"),
        "season": current_season,
        "last_game_date": last_game_date.strftime("%Y-%m-%d"),
        "historical_games": len(history),
        "current_season_games": len(season_history),
        "recent_averages": recent_averages,
    }

    return feature_df, info


def load_model_metadata(model_version):
    model_dir = MODEL_DIRS[model_version]
    path = model_dir / "training_results.json"

    if not path.exists():
        raise FileNotFoundError(
            f"Missing model metadata: {path}"
        )

    with open(path, encoding="utf-8") as file:
        metadata = json.load(file)

    return metadata


def predict_player(
    player_name,
    opponent,
    is_home,
    game_date,
    model_version="v2",
    season=None,
):
    """Generate six stat predictions and combined expected totals."""

    history = load_history()

    feature_df, info = build_prediction_features(
        history,
        player_name,
        game_date,
        is_home,
        season=season,
    )

    metadata = load_model_metadata(model_version)
    feature_columns = metadata["features"]

    missing = [
        col for col in feature_columns
        if col not in feature_df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing {model_version} features: {missing}"
        )

    X = feature_df[feature_columns]

    if X.isna().any().any():
        raise ValueError(
            "Prediction features contain missing values."
        )

    predictions = {}
    model_dir = MODEL_DIRS[model_version]

    for target in TARGETS:
        model_path = (
            model_dir / f"{target.lower()}_model.cbm"
        )

        if not model_path.exists():
            raise FileNotFoundError(
                f"Missing trained model: {model_path}"
            )

        model = CatBoostRegressor()
        model.load_model(str(model_path))

        predicted = float(model.predict(X)[0])

        predictions[target] = round(
            max(predicted, 0.0), 2
        )

    predictions["PRA"] = round(
        predictions["PTS"]
        + predictions["REB"]
        + predictions["AST"], 2
    )

    predictions["PR"] = round(
        predictions["PTS"] + predictions["REB"], 2
    )

    predictions["PA"] = round(
        predictions["PTS"] + predictions["AST"], 2
    )

    predictions["RA"] = round(
        predictions["REB"] + predictions["AST"], 2
    )

    info.update({
        "opponent": opponent.upper(),
        "location": "Home" if is_home else "Away",
        "model_version": model_version,
        "predictions": predictions,
    })

    return info


def display_prediction(result):
    print("\n" + "=" * 65)
    print(
        f"       PARLAY GUARD - "
        f"{result['model_version'].upper()} PREDICTIONS"
    )
    print("=" * 65)

    print(f"Player:          {result['player']}")
    print(f"Opponent:        {result['opponent']}")
    print(f"Location:        {result['location']}")
    print(f"Game date:       {result['game_date']}")
    print(f"Season:          {result['season']}")
    print(f"Last game:       {result['last_game_date']}")
    print(f"History:         {result['historical_games']} games")
    print(f"Season games:    {result['current_season_games']}")

    print("\n" + "-" * 65)
    print(
        f"{'STAT':<10}"
        f"{'PREDICTED':>12}"
        f"{'LAST 5':>12}"
        f"{'LAST 10':>12}"
        f"{'STD 10':>12}"
    )
    print("-" * 65)

    for stat in TARGETS:
        prediction = result["predictions"][stat]
        recent = result["recent_averages"][stat]

        print(
            f"{stat:<10}"
            f"{prediction:>12.2f}"
            f"{recent['last_5']:>12.2f}"
            f"{recent['last_10']:>12.2f}"
            f"{recent['std_10']:>12.2f}"
        )

    print("-" * 65)

    print("\nCOMBINED STAT PREDICTIONS")

    for stat in ["PRA", "PR", "PA", "RA"]:
        print(
            f"{stat:<10}"
            f"{result['predictions'][stat]:>12.2f}"
        )

    print(
        "\nNote: Standard deviation measures recent variability."
    )
    print(
        "Predictions are expected totals, not calibrated"
    )
    print(
        "over/under probabilities or betting recommendations."
    )
    print("=" * 65 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Predict NBA player statistics"
    )

    parser.add_argument(
        "--player", required=True
    )

    parser.add_argument(
        "--opponent", required=True
    )

    location = parser.add_mutually_exclusive_group(
        required=True
    )

    location.add_argument(
        "--home", action="store_true"
    )

    location.add_argument(
        "--away", action="store_true"
    )

    parser.add_argument(
        "--date",
        default=pd.Timestamp.now().strftime("%Y-%m-%d"),
    )

    parser.add_argument(
        "--season",
        help="NBA season, e.g. 2026-27",
    )

    parser.add_argument(
        "--model-version",
        choices=["v1", "v2"],
        default="v2",
    )

    parser.add_argument(
        "--json",
        action="store_true",
    )

    args = parser.parse_args()

    game_date = pd.Timestamp(args.date).normalize()

    result = predict_player(
        player_name=args.player,
        opponent=args.opponent,
        is_home=args.home,
        game_date=game_date,
        model_version=args.model_version,
        season=args.season,
    )

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        display_prediction(result)


if __name__ == "__main__":
    main()
