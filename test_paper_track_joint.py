# # Parlay Guard — paper outcome tracker (research only)

# ## Purpose

# Compare **singles**, **cross-game prop + prop**, **same-game prop + spread/total**, and **cross-game prop + spread/total** over future NBA dates. This is not a bet placer, not proof of an edge, and not a means of producing verified executable parlay ROI. Always compare the strategies over a common, fully observed set of dates before drawing conclusions.

# ## What happens automatically

# After a successful live refresh and Joint Probability Lab calculation:

# 1. A slate is captured for each NBA date that has fresh, qualifying winner-team selected props with unambiguous player IDs and a frozen NBA game ID. One immutable capture per **date**; subsequent refreshes never revise it. A day with no qualifying props is not captured.
# 2. A settlement sidecar is computed from local NBA team-game logs and completed player appearances when they become available. The code requires two correctly paired team results (same NBA game ID, both teams, a winner and valid scores); it also requires a unique, recorded player appearance on that exact NBA game ID and same selected team.
# 3. `paper_tracking` is returned with the live snapshot, shown on the dashboard. It reports ticket hit rate, Brier score, pending tickets, fully settled dates and reference returns where a price reference exists.

# ## File storage

# `data/paper_trades/joint_tracker/snapshots/YYYY-MM-DD.json` are **never modified**. They freeze player IDs, markets, modeled probabilities and observed top asks before all linked games begin.

# `data/paper_trades/joint_tracker/settlements/YYYY-MM-DD.json` are recomputed when more verified results become available; a settlement never changes the original pregame record.

# All files are local and gitignored under the existing `data/paper_trades/` rule. Avoid committing a repository snapshot containing personal or market data unless intended.

# ## CLI

# From the project root:

# ```powershell
# python -m src.paper_track_joint report
# python -m src.paper_track_joint settle
# ```

# Optionally supply local **trusted, complete** files with `--team-csv PATH --player-csv PATH` to `settle`. By default, team results come from `data/nba_YYYY_YY_latest_raw.csv`, and player appearances come from `data/live_player_training_data_v2.csv`. The normal live pipeline updates these, but if your data source is blocked or a player is missing from the engineered overlay, outcomes remain **PENDING** until a trusted completed player game record becomes available. No-show settlement rules, injuries, voids and retroactive corrections are **not** automatically inferred. If team data is missing, no outcomes are settled.

# ## Interpret the results carefully

# - `hit_rate` and Brier score use verified WIN/LOSS tickets only; pending and push cases are excluded.
# - Paper-reference ROI for single props uses the **observed ask**, not a fill. It excludes fees, slippage and other trading costs.
# - Paper-reference ROI for cross-game prop pairs uses the **synthetic product** of two observed asks; it is **NOT an executable Combo price**. Never call it realized betting return.
# - Same-game prop + line and cross-game prop + line tickets have **no captured executable Combo ask** and therefore cannot have an ROI. They are evaluated on joint hit rate / Brier only.
# - Equal-budget reference ROI includes only **fully resolved daily slates**, with equal stakes across that strategy's tickets on a date. Other strategy sets may exist on different dates, so comparing their means is not a controlled test by itself.
# - Model probabilities are **uncalibrated** and correlations estimated from historical data may change. Observed market prices can differ substantially from executable odds.

# **No historical picks are backfilled**. Only slates captured before tipoff are eligible. This prevents accidental evaluation on predictions generated after the games.
