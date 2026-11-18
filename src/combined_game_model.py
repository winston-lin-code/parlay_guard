"""Nested chronological test: Elo game model vs Elo + predicted star risk.

Run from the project root:
    python -u -m src.combined_game_model

Inputs (already produced by other project modules):
    data/nba_model_dataset.csv
    data/nba_games_updated.csv
    data/nba_player_games.csv
    data/nba_scoring_dependency.csv
    data/nba_player_underperformance_dataset.csv

IMPORTANT:
- Player-risk models are retrained on prior games only and predict later blocks.
- Game training receives *historical out-of-sample* player probabilities, not
  probabilities fitted on the same game's star-scoring outcome.
- A star is selected from previous team games. Pregame player features are
  constructed even when the selected star DID NOT play the current game.
  STAR_PLAYED and current-game ACTUAL_PTS are never game-model inputs.
- Both game models are trained and evaluated on identical chronological folds.
- December 29, 2025 and later is never trained on or evaluated here.

Player-risk probabilities are conditional on the selected star playing. Since
we do not model availability, they are an imperfect feature when a star sits.
"""
from collections import defaultdict, deque

import numpy as np
import pandas as pd

from .common import DATA, MODELS
from .leading_scorer_points import FEATURES as PLAYER_FEATURES, HOLDOUT_START, load_data
from .leading_scorer_underperformance import residual_probability
from .walk_forward import make_model, score

RISK_NAME = 'STAR_UNDERPERFORMANCE_PROB'
RISK_COLUMNS = ['HOME_' + RISK_NAME, 'AWAY_' + RISK_NAME]
MIN_PLAYER_TRAIN_ROWS = 400
PLAYER_RETRAIN_EVERY_DAYS = 28


def _read_csv(filename, **kwargs):
    path = DATA / filename
    if not path.exists():
        raise FileNotFoundError(f'Missing {path}. Run the existing data-building scripts first.')
    return pd.read_csv(path, dtype={'GAME_ID': str}, **kwargs)


def _bool_column(series, name):
    values = series.astype(str).str.strip().str.lower().map({
        'true': True, 'false': False, '1': True, '0': False,
    })
    if values.isna().any():
        raise ValueError(f'{name} has unrecognized Boolean values')
    return values.astype(bool)


def selected_stars_pregame_features():
    """Rebuild *every* preselected star's inputs, irrespective of today's play.

    Uses only strictly earlier dates for player and opponent rolling histories.
    Each day's scores/appearances are appended only AFTER all its rows are made.
    No STAR_PLAYED/current box score is consulted to decide a row's presence.
    """
    players = _read_csv('nba_player_games.csv')
    teams = _read_csv('nba_games_updated.csv')
    stars = _read_csv('nba_scoring_dependency.csv')
    required_p = {'GAME_ID', 'GAME_DATE', 'TEAM_ID', 'PLAYER_ID', 'PTS', 'MIN', 'FGA'}
    required_t = {'GAME_ID', 'GAME_DATE', 'TEAM_ID', 'PTS', 'IS_HOME'}
    required_s = {'GAME_ID', 'TEAM_ID', 'STAR_PLAYER_ID', 'TOP_SCORER_SHARE_L10'}
    for label, frame, required in [('player', players, required_p),
                                   ('team', teams, required_t),
                                   ('star selection', stars, required_s)]:
        if required - set(frame):
            raise ValueError(f'Missing {label} columns: {sorted(required-set(frame))}')
    for df in (players, teams, stars):
        df['GAME_ID'] = df['GAME_ID'].astype(str).str.zfill(10)
        df['TEAM_ID'] = pd.to_numeric(df['TEAM_ID'], errors='raise').astype('int64')
    for df in (players, teams):
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'], errors='raise').dt.normalize()
        df.drop(df.index[df.GAME_DATE >= HOLDOUT_START], inplace=True)
    players['PLAYER_ID'] = pd.to_numeric(players['PLAYER_ID'], errors='raise').astype('int64')
    stars['STAR_PLAYER_ID'] = pd.to_numeric(stars['STAR_PLAYER_ID'], errors='raise').astype('int64')
    for col in ('PTS', 'MIN', 'FGA'):
        players[col] = pd.to_numeric(players[col], errors='coerce')
    teams['PTS'] = pd.to_numeric(teams['PTS'], errors='raise')
    teams['IS_HOME'] = _bool_column(teams['IS_HOME'], 'IS_HOME')
    if teams.duplicated(['GAME_ID', 'TEAM_ID']).any():
        raise ValueError('Duplicate team-game rows')
    if players.duplicated(['GAME_ID', 'TEAM_ID', 'PLAYER_ID']).any():
        raise ValueError('Duplicate player-game rows')
    if stars.duplicated(['GAME_ID', 'TEAM_ID']).any():
        raise ValueError('Duplicate star selection rows')

    # Build opponent IDs from the scheduled two teams, not from the result.
    opponent = {}
    for gid, group in teams.groupby('GAME_ID', sort=False):
        if len(group) != 2 or group.TEAM_ID.nunique() != 2:
            raise ValueError(f'Expected two different teams in game {gid}')
        t0, t1 = map(int, group.TEAM_ID.to_numpy())
        opponent[(gid, t0)] = t1
        opponent[(gid, t1)] = t0

    # Date comes from the team-game schedule, never from postgame star fields.
    selected = stars[['GAME_ID', 'TEAM_ID', 'STAR_PLAYER_ID',
                      'TOP_SCORER_SHARE_L10']].merge(
        teams[['GAME_ID', 'TEAM_ID', 'GAME_DATE', 'IS_HOME']],
        on=['GAME_ID', 'TEAM_ID'], how='inner', validate='one_to_one'
    )
    selected['TOP_SCORER_SHARE_L10'] = pd.to_numeric(
        selected['TOP_SCORER_SHARE_L10'], errors='coerce')
    selected_by_date = dict(tuple(selected.groupby('GAME_DATE', sort=True)))
    players_by_date = dict(tuple(players.groupby('GAME_DATE', sort=True)))
    player_history = defaultdict(lambda: deque(maxlen=10))
    team_history = defaultdict(lambda: deque(maxlen=10))
    rows = []

    teams = teams.sort_values(['GAME_DATE', 'GAME_ID', 'TEAM_ID'])
    for date, todays_teams in teams.groupby('GAME_DATE', sort=True):
        if date in selected_by_date:
            for s in selected_by_date[date].itertuples(index=False):
                gid, tid, pid = str(s.GAME_ID), int(s.TEAM_ID), int(s.STAR_PLAYER_ID)
                past = list(player_history[pid])
                if len(past) < 10:
                    continue   # Known pregame lack of prior player appearances.
                last5 = past[-5:]
                opp_past = list(team_history[opponent[(gid, tid)]])
                own_past = list(team_history[tid])
                points = np.asarray([x['points'] for x in past], dtype=float)
                minutes = np.asarray([x['minutes'] for x in past], dtype=float)
                fga = np.asarray([x['fga'] for x in past], dtype=float)
                opp_allowed = [x['points_allowed'] for x in opp_past]
                own_scored = [x['points_scored'] for x in own_past]
                days_rest = int((date - past[-1]['date']).days - 1)
                if days_rest < 0:
                    raise ValueError(f'Negative rest for {pid} on {date}')
                rows.append({
                    'GAME_ID': gid, 'GAME_DATE': date, 'TEAM_ID': tid,
                    'STAR_PLAYER_ID': pid,
                    'TOP_SCORER_SHARE_L10': float(s.TOP_SCORER_SHARE_L10),
                    'IS_HOME': int(s.IS_HOME),
                    'PLAYER_PPG_LAST_5': float(np.mean([x['points'] for x in last5])),
                    'PLAYER_PPG_LAST_10': float(points.mean()),
                    'PLAYER_PTS_STD_LAST_10': float(points.std(ddof=0)),
                    'PLAYER_MIN_LAST_5': float(np.nanmean([x['minutes'] for x in last5])),
                    'PLAYER_MIN_LAST_10': float(np.nanmean(minutes)),
                    'PLAYER_FGA_LAST_5': float(np.nanmean([x['fga'] for x in last5])),
                    'PLAYER_FGA_LAST_10': float(np.nanmean(fga)),
                    'PLAYER_DAYS_REST': days_rest,
                    'OPP_PTS_ALLOWED_LAST_5': (float(np.mean(opp_allowed[-5:]))
                                                if len(opp_allowed) >= 5 else np.nan),
                    'OPP_PTS_ALLOWED_LAST_10': (float(np.mean(opp_allowed))
                                                 if len(opp_allowed) >= 10 else np.nan),
                    'TEAM_PTS_SCORED_LAST_10': (float(np.mean(own_scored))
                                                 if len(own_scored) >= 10 else np.nan),
                })

        # Add today's appearances only after generating ALL of today's features.
        todays_players = players_by_date.get(date)
        if todays_players is not None:
            for p in todays_players.itertuples(index=False):
                minutes = float(p.MIN) if pd.notna(p.MIN) else np.nan
                points = float(p.PTS) if pd.notna(p.PTS) else np.nan
                fga = float(p.FGA) if pd.notna(p.FGA) else np.nan
                if not np.isfinite(minutes) or minutes <= 0 or not np.isfinite(points):
                    continue
                player_history[int(p.PLAYER_ID)].append({
                    'date': date, 'points': points, 'minutes': minutes, 'fga': fga,
                })

        scores = {(str(t.GAME_ID), int(t.TEAM_ID)): float(t.PTS)
                  for t in todays_teams.itertuples(index=False)}
        for t in todays_teams.itertuples(index=False):
            gid, tid = str(t.GAME_ID), int(t.TEAM_ID)
            team_history[tid].append({
                'points_scored': scores[(gid, tid)],
                'points_allowed': scores[(gid, opponent[(gid, tid)])],
            })

    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError('No eligible pregame star features were constructed')
    if result.duplicated(['GAME_ID', 'TEAM_ID']).any():
        raise ValueError('Multiple star feature rows per team-game')
    return result.sort_values(['GAME_DATE', 'GAME_ID', 'TEAM_ID']).reset_index(drop=True), len(selected)


def verify_against_player_dataset(star_train, pregame):
    """Guard against accidental changes to the previously evaluated inputs."""
    keys = ['GAME_ID', 'TEAM_ID', 'STAR_PLAYER_ID']
    joined = star_train[keys + PLAYER_FEATURES].merge(
        pregame[keys + PLAYER_FEATURES], on=keys, how='left',
        suffixes=('_previous', '_rebuilt'), validate='one_to_one', indicator=True
    )
    missing = int((joined['_merge'] != 'both').sum())
    if missing:
        raise AssertionError(f'{missing} played-star samples have no rebuilt pregame features')
    for feature in PLAYER_FEATURES:
        a = pd.to_numeric(joined[feature + '_previous'], errors='coerce').to_numpy(dtype=float)
        b = pd.to_numeric(joined[feature + '_rebuilt'], errors='coerce').to_numpy(dtype=float)
        matched = np.isclose(a, b, atol=1e-6, rtol=1e-6, equal_nan=True)
        if not matched.all():
            raise AssertionError(f'{feature} differs for {int((~matched).sum())} '
                                 'rows between existing player dataset and pregame rebuild')
    print(f'Verified {len(joined):,} played-star rows against prior pregame features.', flush=True)


def generate_historical_star_risk(star_train, pregame):
    """Fit prior-only Ridge models, predict subsequent 28-day blocks.

    Training uses ACTUAL_PTS only before each prediction block. Risk is generated
    for ALL preselected stars with 10 prior appearances, not just stars who played.
    The first dates have insufficient prior training; their risks remain missing.
    """
    star_train = star_train.loc[star_train.GAME_DATE < HOLDOUT_START].sort_values('GAME_DATE')
    pregame = pregame.loc[pregame.GAME_DATE < HOLDOUT_START].sort_values('GAME_DATE')
    if star_train.empty or pregame.empty:
        raise ValueError('Empty pre-holdout player data')
    dates = np.array(sorted(pregame.GAME_DATE.unique()))
    first_start = None
    for d in dates:
        if int((star_train.GAME_DATE < d).sum()) >= MIN_PLAYER_TRAIN_ROWS:
            first_start = pd.Timestamp(d)
            break
    if first_start is None:
        raise ValueError('Not enough historical star appearances for first player-risk model')
    print(f'First prior-only star-risk forecast: {first_start.date()} '
          f'(minimum historical star samples: {MIN_PLAYER_TRAIN_ROWS:,})', flush=True)

    risk_frames = []
    start = first_start
    block = 0
    while start < HOLDOUT_START:
        end = min(start + pd.Timedelta(days=PLAYER_RETRAIN_EVERY_DAYS), HOLDOUT_START)
        past = star_train.loc[star_train.GAME_DATE < start]
        future_block = pregame.loc[(pregame.GAME_DATE >= start) & (pregame.GAME_DATE < end)]
        if not future_block.empty:
            if len(past) < MIN_PLAYER_TRAIN_ROWS:
                raise AssertionError('Insufficient samples at forecasting date')
            # This imports the SAME Ridge+empirical-residual-CDF technique that
            # won the standalone star experiment, with no future outcomes.
            risk, calibration_n, _ = residual_probability(past, future_block)
            if not np.isfinite(risk).all() or ((risk <= 0) | (risk >= 1)).any():
                raise ValueError('Invalid star probability estimates')
            r = future_block[['GAME_ID', 'GAME_DATE', 'TEAM_ID', 'STAR_PLAYER_ID']].copy()
            r[RISK_NAME] = risk
            r['forecast_train_end_exclusive'] = start
            r['forecast_calibration_rows'] = calibration_n
            risk_frames.append(r)
        block += 1
        if block == 1 or block % 5 == 0 or end == HOLDOUT_START:
            print(f'  Star-risk forecast block {block}: dates [{start.date()}, {end.date()}) '
                  f'past played-star rows={len(past):,}, '
                  f'forecasted team-games={len(future_block):,}', flush=True)
        start = end

    if not risk_frames:
        raise ValueError('No star risk predictions were generated')
    result = pd.concat(risk_frames, ignore_index=True)
    if result.duplicated(['GAME_ID', 'TEAM_ID']).any():
        raise AssertionError('Duplicate risk per team-game')
    # The model fitting cutoff is always no later than the forecast game.
    # Training itself uses GAME_DATE < this cutoff (strictly earlier dates).
    if not (result.forecast_train_end_exclusive <= result.GAME_DATE).all():
        raise AssertionError('Player forecasting chronology was violated')
    return result


def add_risk_to_games(games, risk, teams):
    # Map by the scheduled home/away team; NEVER use STAR_PLAYED/current PTS.
    mapping = teams[['GAME_ID', 'TEAM_ID', 'IS_HOME']].copy()
    mapping['IS_HOME'] = _bool_column(mapping['IS_HOME'], 'IS_HOME')
    if mapping.duplicated(['GAME_ID', 'TEAM_ID']).any():
        raise ValueError('Duplicate home/away mapping')
    if mapping.groupby('GAME_ID')['IS_HOME'].agg(['sum', 'count']).ne(
        {'sum': 1, 'count': 2}
    ).any().any():
        raise ValueError('Expected exactly one home and one away team per game')
    mapped = risk.merge(mapping, on=['GAME_ID', 'TEAM_ID'], validate='one_to_one')
    output = games.copy()
    for side, is_home in [('HOME', True), ('AWAY', False)]:
        side_risk = mapped.loc[mapped.IS_HOME == is_home, ['GAME_ID', RISK_NAME]].rename(
            columns={RISK_NAME: f'{side}_{RISK_NAME}'})
        if side_risk.GAME_ID.duplicated().any():
            raise ValueError(f'Duplicate {side} team risk per game')
        output = output.merge(side_risk, on='GAME_ID', how='left', validate='one_to_one')
    return output


def game_model(with_star_risk):
    model, inputs = make_model(include_team_identity=False, include_point_diff=True,
                               elo_mode='difference')
    if with_star_risk:
        pre = model.named_steps['preprocessor']
        numeric_name, numeric_pipeline, numeric_cols = pre.transformers[0]
        pre.transformers = [(numeric_name, numeric_pipeline,
                             list(numeric_cols) + RISK_COLUMNS)]
        inputs = list(inputs) + RISK_COLUMNS
    return model, inputs


def main():
    print('===== COMBINED ELO + OUT-OF-SAMPLE STAR RISK EXPERIMENT =====', flush=True)
    games = _read_csv('nba_model_dataset.csv')
    teams = _read_csv('nba_games_updated.csv')
    games['GAME_ID'] = games['GAME_ID'].astype(str).str.zfill(10)
    teams['GAME_ID'] = teams['GAME_ID'].astype(str).str.zfill(10)
    games['GAME_DATE'] = pd.to_datetime(games['GAME_DATE'], errors='raise').dt.normalize()
    teams['GAME_DATE'] = pd.to_datetime(teams['GAME_DATE'], errors='raise').dt.normalize()
    if games.GAME_ID.duplicated().any():
        raise ValueError('Duplicate game rows in nba_model_dataset.csv')
    games = games.sort_values(['GAME_DATE', 'GAME_ID']).reset_index(drop=True)
    dates = np.array(sorted(games.GAME_DATE.unique()))
    if len(dates) < 30:
        raise ValueError('Not enough distinct dates for walk-forward validation')
    boundaries = [int(len(dates) * f) for f in (0.40, 0.55, 0.70, 0.85)]
    if len(set(boundaries)) != 4:
        raise ValueError('Date boundaries overlap')
    test_start = pd.Timestamp(dates[boundaries[-1]])
    if test_start != HOLDOUT_START:
        raise ValueError(f'Holdout mismatch: dataset implies {test_start.date()}, '
                         f'but project player models use {HOLDOUT_START.date()}. '
                         'Do not run mismatched validation periods.')
    print(f'Final holdout begins {HOLDOUT_START.date()} — NOT evaluated.', flush=True)
    print('Building pregame features for the previously selected star, whether or not they play...',
          flush=True)
    pregame, n_selected = selected_stars_pregame_features()
    print(f'Pregame star features: {len(pregame):,} team-games from '
          f'{n_selected:,} selections. No current-game availability was used.', flush=True)

    star_train, unmatched = load_data()
    star_train = star_train.loc[star_train.GAME_DATE < HOLDOUT_START].copy()
    print(f'Historical star training appearances: {len(star_train):,} '
          f'(unmatched from previous join across all dates: {unmatched:,})', flush=True)
    verify_against_player_dataset(star_train, pregame)
    risk = generate_historical_star_risk(star_train, pregame)
    dev = add_risk_to_games(games.loc[games.GAME_DATE < HOLDOUT_START].copy(),
                            risk, teams)
    print(f'Predicted star risks for {len(risk):,} historical team-games.', flush=True)
    print('Missing risk features by game folds:', flush=True)

    rows, out_predictions = [], []
    for i in range(3):
        start = pd.Timestamp(dates[boundaries[i]])
        end = pd.Timestamp(dates[boundaries[i + 1]])
        train = dev.loc[dev.GAME_DATE < start]
        valid = dev.loc[(dev.GAME_DATE >= start) & (dev.GAME_DATE < end)]
        if len(train) < 100 or len(valid) < 100:
            raise ValueError(f'Fold {i+1} has insufficient game rows')
        if set(train.GAME_ID) & set(valid.GAME_ID):
            raise AssertionError('A game is in both training and validation')
        print(f'\nFold {i+1}/3: train={len(train):,}, validate={len(valid):,} '
              f'[{start.date()}, {end.date()})', flush=True)
        for col in RISK_COLUMNS:
            count = int(valid[col].isna().sum())
            print(f'  {col}: {count:,}/{len(valid):,} validation values missing', flush=True)

        pred_by_model = {}
        for name, with_risk in [('elo_baseline', False), ('elo_plus_star_risk', True)]:
            model, inputs = game_model(with_star_risk=with_risk)
            missing = set(inputs) - set(dev.columns)
            if missing:
                raise ValueError(f'Missing game model inputs: {sorted(missing)}')
            model.fit(train[inputs], train.HOME_WIN.astype(int))
            p = model.predict_proba(valid[inputs])[:, 1]
            metrics = score(valid.HOME_WIN.astype(int), p)
            rows.append({'fold': i + 1, 'model': name, 'train_games': len(train),
                         'validation_games': len(valid), 'validation_start': start.date(),
                         'validation_end_exclusive': end.date(), **metrics})
            pred_by_model[name] = p
            print(f'  {name:22s} log_loss={metrics["log_loss"]:.5f} '
                  f'brier={metrics["brier"]:.5f} accuracy={metrics["accuracy"]:.2%} '
                  f'roc_auc={metrics["roc_auc"]:.4f}', flush=True)

        out_predictions.append(pd.DataFrame({
            'fold': i + 1,
            'GAME_ID': valid.GAME_ID.to_numpy(),
            'GAME_DATE': valid.GAME_DATE.to_numpy(),
            'HOME_WIN': valid.HOME_WIN.astype(int).to_numpy(),
            'elo_baseline_prob': pred_by_model['elo_baseline'],
            'elo_plus_star_risk_prob': pred_by_model['elo_plus_star_risk'],
            RISK_COLUMNS[0]: valid[RISK_COLUMNS[0]].to_numpy(),
            RISK_COLUMNS[1]: valid[RISK_COLUMNS[1]].to_numpy(),
        }))

    folds = pd.DataFrame(rows)
    summary = (folds.groupby('model', as_index=False)
               .agg(mean_log_loss=('log_loss', 'mean'),
                    mean_brier=('brier', 'mean'),
                    mean_accuracy=('accuracy', 'mean'),
                    mean_roc_auc=('roc_auc', 'mean'),
                    validation_games=('validation_games', 'sum'))
               .sort_values('mean_log_loss'))
    print('\n===== COMBINED GAME MODEL WALK-FORWARD SUMMARY =====')
    print(summary.to_string(index=False, float_format=lambda x: f'{x:.5f}'))
    comparison = folds.pivot(index='fold', columns='model', values='log_loss')
    print('\n===== PER-FOLD GAME LOG LOSS =====')
    print(comparison.to_string(float_format=lambda x: f'{x:.5f}'))
    delta = comparison['elo_plus_star_risk'] - comparison['elo_baseline']
    print('\nSTAR RISK MINUS ELO BASELINE (negative is better):')
    print(delta.to_string(float_format=lambda x: f'{x:+.5f}'))
    print(f'Improved {int((delta < 0).sum())}/3 folds; '
          f'mean delta={delta.mean():+.5f}', flush=True)
    print('The validation games are identical for both models.')
    print('Star risks are out-of-sample relative to each game; availability is NOT known.')
    print(f'FINAL HOLDOUT {HOLDOUT_START.date()} AND LATER: NOT EVALUATED.', flush=True)

    MODELS.mkdir(parents=True, exist_ok=True)
    folds.to_csv(MODELS / 'combined_game_model_folds.csv', index=False)
    summary.to_csv(MODELS / 'combined_game_model_summary.csv', index=False)
    pd.concat(out_predictions, ignore_index=True).to_csv(
        MODELS / 'combined_game_model_validation_predictions.csv', index=False)
    risk.to_csv(MODELS / 'combined_game_model_historical_star_risk.csv', index=False)
    print('Saved four CSV reports in models/.', flush=True)


if __name__ == '__main__':
    main()
