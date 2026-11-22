# Joint Probability Lab (research only)

This add-on compares Parlay Guard's currently screened **winner-team** player props across four structures:

1. **Singles** — displays the existing model's stressed per-prop return estimate, using a five-percentage-point probability haircut and a hypothetical $0.01 per-share buffer. This is not validated profitability.
2. **Cross-game prop pairs** — estimates joint probability as the product of stressed marginal probabilities. This makes a **conditional independence assumption**; the individual asks are multiplied only to create a **synthetic illustrative** price. There is no verified executable combined price or EV.
3. **Cross-game prop + spread/total** — pairs a winner-team prop from one matchup with a spread/total line from another matchup using a labeled independence assumption; there is no executable line or combo quote, so no pricing EV.
4. **Same-game player prop + spread/total** — estimates spread-cover or total-side probability using a normal approximation with standard deviation estimated from historical holdout MAE. Correlation is estimated from past same-player pregame stat residuals and a rolling team margin/total residual, with 25 minimum paired historical games, 100-game maximum window, shrinkage and Fisher uncertainty. A Gaussian copula combines the two marginal event probabilities; when correlations are missing, joint probability is **unavailable**, not presumed independent.

## Running

Once installed, run the existing local application:

```powershell
python -m pytest tests/ -v --tb=short
python -m uvicorn src.api:app --host 127.0.0.1 --port 8000
```

Request a refresh in a second PowerShell window:

```powershell
Invoke-RestMethod -Method Post http://127.0.0.1:8000/api/live/refresh
$live = Invoke-RestMethod http://127.0.0.1:8000/api/live/data
$live.joint_research | ConvertTo-Json -Depth 6
```

The live page shows the results in **Joint probability lab** after a completed refresh.

## Do not mistake a research estimate for a bet

- **No verified Polymarket Combo ask:** `executable_combo_ask` and `executable_ev_roi` are always `null`. Even a synthetic positive-EV value is not executable.
- **No calibrated spread/total cover probabilities:** approximate probabilities use normal residual errors inferred from MAE, not a dedicated chronological probability calibration study. Do not trust apparent spread/total edges.
- **Correlation uncertainty:** using team line residuals and player stat residuals is only an approximation; same-game correlation may differ by roster, matchup, injuries, pace, and player minutes. The sample and Fisher sensitivity range are diagnostics, not validated confidence intervals for actual market outcomes.
- **No actual injury/availability verification.** A player missing a game can change the settlement result.
- **No realized ROI:** Long-term profitability requires saving immutable pregame snapshots, obtaining real executable quote/depth and settlement information, then evaluating a chronological untouched test period. This feature doesn't claim any strategy is safer or more profitable.
- **Research UI is additive:** Existing illustrative strategy-payout simulator continues to use synthetic combination prices, explicitly labeled non-executable.

## Next research milestones

- Add live venue-supported Combo RFQ executable ask/depth and fee validation before any combo EV claim.
- Backtest and calibrate game-line probabilities chronologically with predicted score residuals for each market line; report Brier/log-loss and reliability curves.
- Estimate same-game correlations on held-out seasons and reject sparse or unstable correlation estimates.
- Maintain a fair **equal-stake**, pregame-frozen paper-trading ledger for singles, cross-game and same-game strategies; settle only from verified source results; compare realized ROI, hit rate, drawdown and confidence intervals.
- Improve source parsing for roster assignment, market resolution and invalid/no-play situations.
