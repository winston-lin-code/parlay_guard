# Parlay Guard — College Moneyline Models and EV

This is a **read-only research extension** of an existing Parlay Guard installation.

## What's included

- **Men's NCAA Division I basketball (MBB)** and **NCAA FBS football (CFB)**.
- Separate chronological-model fits using completed results from public SportsDataverse releases based on ESPN schedules. The training command downloads **real historical outcomes** and fits sklearn logistic regressions on leakage-controlled pregame Elo, form, rest, venue, and game-count features. It reports **chronological 20% holdout** Brier score, log loss, accuracy, and a baseline Brier score. No default/dummy probabilities are ever substituted for missing models.
- College games and official ESPN school logos from ESPN scoreboards (`groups=50` MBB and `groups=80` CFB).
- Strictly mapped two-team moneyline markets for separate league series on Polymarket.com (MBB series 10470, CFB 10210); requires matching both team names (or unique opponent-specific abbreviations) and kickoff time within 2h. Market prices are **actual observed best YES asks**, not Gamma indicative midpoints. Ambiguous mappings/unknown team IDs are excluded.
- Both sides' estimated EV: `model_probability / (ask + $0.01/share illustrative fee buffer) - 1`. Display both price and probabilities; positive label at >=5% *estimated* EV. Price and fee assumptions do NOT establish real profitability.
- ESPN injury feed (where populated), league-team roster listings with changes since the last check, and optional attributable local roster reports. **An unpopulated or inaccessible feed means UNVERIFIED**, not "healthy". Injuries and roster-list deltas are **displayed alongside the matchup** but are **not inputs to the trained model**; games with positive estimated EV and a reported issue are marked for review.
- `/college` full game dashboard; main NBA dashboard has a college preview and link. `date_offset=0,1,2` allows today, tomorrow, and the next day.
- Offline unit/integration tests. Existing NBA and frozen moneyline/same-game files are never edited.

## Install (Windows)

1. Stop Uvicorn (`Ctrl+C`), unzip the package **inside** your existing Parlay Guard directory, beside `src`, `static` and `models`.
2. Double-click `START_COLLEGE_GUARD.cmd`. It runs the idempotent installer, installs PyArrow if needed, downloads real college historical seasons, fits two models, runs tests, and starts Uvicorn.
3. Open `http://127.0.0.1:8000/college` or the **College sports** section on the NBA home page.

Manual equivalent in PowerShell:

```powershell
cd C:\Users\Winston\Desktop\parlay_guard_project\parlay_guard
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8 = '1'
python install_college_guard.py
python -m pip install pyarrow
python -m src.college_train --sports mbb,cfb --refresh-current
python -m pytest tests/test_college_guard.py -q
$env:PG_MAX_GAMES='20'
python -m uvicorn src.api:app --reload --host 127.0.0.1 --port 8000
```

**The model training downloads may take time and require internet access to GitHub release assets.** Check the printed `FIT COMPLETED` and out-of-sample metrics for each sport. If one season release is absent, the downloader skips it; if insufficient real results remain, it **does not create a model**. The dashboard indicates which league is untrained. If PyArrow cannot install or your computer cannot reach GitHub releases, use `python -m src.college_train` when those dependencies are available. No claim of pretraining is made for the ZIP itself.

SportsDataverse schedules: `https://github.com/sportsdataverse/sportsdataverse-data/releases/tag/espn_mens_college_basketball_schedules` and `https://github.com/sportsdataverse/sportsdataverse-data/releases/tag/cfb_schedules`.

### Optional authoritative/verified roster reports

Place `data/college_guard/roster_reports.csv` with header:

` sport,team_id,player,status,checked_at_utc,source_url `

Example format (not a real injury):

`mbb,123,Example Athlete,Questionable,2026-11-21T12:00:00+00:00,https://example.edu/source`

Entries require a source URL and become stale after 36 hours. A roster change indicates the **ESPN listing differs**; it must not be interpreted as a confirmed injury, suspension or transfer.

### Controls

- `PG_COLLEGE_MAX_GAMES`: maximum displayed games per sport (default `55`)
- `PG_COLLEGE_ROSTER_TEAMS`: maximum unique teams per sport checked on each 10-minute refresh (default `28`). Others are marked `not checked`; not all college programs publish an accessible injury feed.
- `PG_COLLEGE_MIN_EV`: threshold to highlight, default `0.05` (5%).
- `PG_COLLEGE_FEE_BUFFER`: extra dollars per share in EV formula, default `0.01`. The actual venue fees vary and must be checked independently.

### Important limitations

- This **does not place orders or provide Polymarket US execution**. Polymarket.com and Polymarket US have separate contracts and prices; don't treat these quotes as eligible executable quotes from the United States.
- The models are **new research baselines**. The offline Brier/log loss/accuracy measures are not profitable-betting backtests; without historical executable Polymarket odds, no proof of +EV/profitability exists. There is no guarantee that the training models outperform the markets.
- College transfer portal changes, player availability, suspensions and injuries can be missing; ESPN feeds are not authoritative for every school. Injury flags **do not currently adjust predicted probabilities**. Treat discrepancies as reasons for additional verification.
- Preseason team rankings, lineups, injuries, and roster continuity are *not yet* part of the training feature set. Year-over-year Elo is regressed toward 1500 by 35% per team when the new season begins.
- Original NBA outputs, frozen strategy checksum, paper trade logs, the same-game comparison, and bankroll allocations are untouched.
