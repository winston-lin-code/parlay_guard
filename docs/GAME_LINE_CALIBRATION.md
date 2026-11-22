# Parlay Guard — NBA game-line probability calibration research

## What it does

Run `python -m src.calibrate_game_lines` from the project root after refreshing the live NBA historical team cache. This is an offline research command: it **does not** fetch market quotes, place orders, or change the winner model, player-prop model, or stored paper-trading snapshots.

It reuses the same `src.game_lines_live.feature_training` features and Ridge hyperparameter (`alpha=150`) as the existing spread/total predictor. Games are grouped by date and use only earlier completed games to create pregame features.

- **Train:** all feature-ready games before October 1, 2024.
- **Validation/calibration:** October 1, 2024 through September 30, 2025. Fit a probability map to outcomes observed on synthetic thresholds around the forecast.
- **Untouched test:** October 1, 2025 through June 30, 2026. Report baseline vs calibrated Brier score, 10-bin ECE and reliability bins on the **test only**.
- Fixed sigma is derived from *validation* absolute errors. Isotonic mapping is fitted on the validation set only. Test results do **not** fit the calibrator.

The model fits on pre-2024-25 games, so the evaluated predictor is **not identical** to the current live model, which retrains using all completed games. Also, the thresholds are synthetic offsets relative to each forecast; **they are not historic sportsbook spreads/totals or Polymarket fills**. The test includes multiple correlated events per game, which should not be treated as independent samples. Good Brier/ECE on this threshold grid does not establish market-line profitability.

The calibration JSON is written under `models/game_line_calibration_v1.json`. It uses a safe declarative JSON format, not pickle. It includes split sizes and dates, isotonic knots, and the fixed, predeclared acceptance checks:

- Calibration test Brier must not exceed baseline by more than 0.002.
- Calibration test ECE must not exceed baseline by more than 0.010.
- Calibration test ECE must not exceed 0.075.

A failed gate means the corresponding spread or total continues using the **uncalibrated normal approximation** in the Joint Probability Lab. Where an offered threshold falls outside the synthetic validation range, the system also declines extrapolation and reports `outside_calibration_support`. Only passing targets use test-evaluated synthetic-threshold calibration, and are labeled accordingly. **Market-line validation remains false**, even when a gate passes. Actual executable combo EV remains unavailable.

The integration updates the Joint Probability Lab's marginal game-line probability estimate, preserving explicit reverse-side (away spread and under-total) complements, and stores provenance flags in both same-game and cross-game research. Existing recorded paper snapshots remain immutable and are not retroactively revised.

## Installation

Extract the upgrade ZIP in the Parlay Guard project root, then run:

```powershell
python .\install_line_calibration.py
python -m pytest tests/ -q
python -m src.calibrate_game_lines
```

Restart Uvicorn and trigger a live refresh. In the live API inspect:

```powershell
$live = Invoke-RestMethod http://127.0.0.1:8000/api/live/data
$live.joint_research.line_calibration | ConvertTo-Json -Depth 6
```

No calibration dataset exists inside this package. The command evaluates the cached team data **on your machine** and may refuse to fit if the chronological splits do not contain enough games. Don't bypass that check simply to force an EV estimate.

## Future expansion

Collect timestamped historical **actual offered** spreads and totals, paired with pregame forecasts and verified final scores. With enough data, use an independent, chronological market-line evaluation to determine whether probability estimates calibrated on synthetic thresholds transfer to real quotes. Evaluate seasonal drift and model recalibration before any deployment claim.
