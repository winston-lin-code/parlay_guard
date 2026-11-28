"""Fit the all-feature WCBB logistic model and build its as-of season context."""

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import NamedTemporaryFile
import json

import pandas as pd
from threadpoolctl import threadpool_limits

from . import wcbb_feature_experiments as research
from .env_config import env_value
from .wcbb_all_features import DEFAULT_PATH, MODEL_ID, SEASON


def build(data, threads):
    membership = research.read_frame(data / "division_i_teams.csv", ("season", "team_id"))
    games, grids, venues = [], [], {}
    cutoff = pd.Timestamp.now(tz="UTC")
    historical = defaultdict(Counter)
    for season in (*research.SEASONS, SEASON):
        members = set(membership.loc[membership.season.eq(season), "team_id"].astype(int))
        if not members:
            raise ValueError(f"Missing Division I membership for season ending {season}")
        schedule = research.read_season(data, "wbb_schedule", season, research.SCHEDULE)
        plan = research.season_plans(schedule, season, members, historical[season])
        boxes = research.box_possessions(research.read_season(data, "team_box", season, research.BOX),
                                        plan, historical[season])
        local, grid = research.snapshots_for_season(
            research.read_season(data, "play_by_play", season, research.PLAYS), plan, boxes, historical[season])
        if not local:
            raise ValueError(f"No audited games for season ending {season}")
        if season == SEASON:
            # Prospective results update pregame context only after release.
            local = [game for game in local if pd.notna(game["available_at"]) and game["available_at"] < cutoff]
            if not local:
                raise ValueError(f"No released games for season ending {season}")
            known = schedule.assign(tip=research.timestamps(schedule.start_date))
            known["home_venue_id"] = pd.to_numeric(known.home_venue_id, errors="coerce")
            known = known[(known.tip < cutoff) & known.home_id.isin(members)]
            known = known.dropna(subset=["home_venue_id"]).sort_values("tip")
            venues = {str(int(row.home_id)): str(int(row.home_venue_id))
                      for row in known.itertuples() if row.home_venue_id > 0}
        else:
            grids.extend(grid)
        games.extend(local)
        print(f"Season {season}: {len(local):,} audited games", flush=True)

    counts = Counter(game["game_id"] for game in games)
    games = [game for game in games if counts[game["game_id"]] == 1]
    historical_elo, context = {}, defaultdict(list)

    def released(at, season, team, rating, offense, defense, sos):
        if season < SEASON:
            historical_elo[str(team)] = rating
        else:
            context[str(team)].append([at.timestamp(), rating, offense, defense, sos])

    priors = research.pregame_history(games, on_release=released, through=cutoff)
    names = research.FEATURES["combined_plus_all"]
    training = priors.loc[priors.season.lt(SEASON), ["game_id", "home_win", *[c for c in names if c not in research.SCORE]]]
    grid = pd.concat(grids, ignore_index=True)
    fit = grid.merge(training, on="game_id", validate="many_to_one")
    if fit.empty or fit.home_win.nunique() != 2:
        raise ValueError("Audited training data need both results")
    with threadpool_limits(limits=threads):
        model = research.make_model("logistic").fit(fit[list(names)], fit.home_win)
    imputer, scaler, logistic = (component for _, component in model.steps)
    return {
        "model": MODEL_ID, "version": 1, "season": SEASON, "trained_seasons": list(research.SEASONS),
        "training_games": int(training.game_id.nunique()), "generated_utc": cutoff.isoformat(),
        "features": list(names), "median": imputer.statistics_.tolist(), "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(), "intercept": float(logistic.intercept_[0]),
        "weights": logistic.coef_[0].tolist(), "members": sorted(members),
        "historical_elo": historical_elo, "context": dict(context), "home_venues": venues,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(env_value("WCBB_HISTORY_DIR") or "data/wcbb-history"))
    parser.add_argument("--output", type=Path, default=Path(env_value("WCBB_ANYTIME_MODEL") or DEFAULT_PATH))
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    temporary = None
    try:
        artifact = build(args.data, args.threads)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=args.output.parent, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(artifact, stream, allow_nan=False, separators=(",", ":"))
        temporary.replace(args.output)
        print(f"Exported {artifact['training_games']:,} training games to {args.output.resolve()}")
    except (OSError, ValueError, ImportError, KeyError) as exc:
        parser.exit(2, f"WCBB model export stopped: {exc}\n")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
