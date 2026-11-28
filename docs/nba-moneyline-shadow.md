# NBA moneyline paper study

Run the study from the Parlay Guard project folder while regular-season NBA
games are in progress. The runner uses public ESPN and Polymarket US data.

```powershell
python -m src.nba_moneyline_shadow --latency-seconds 2
```

Each game can contribute its first qualifying anytime signal and its first
qualifying signal from five game-clock minutes before halftime through five
minutes after halftime. Both use the ESPN play forecast, a win probability
above 60%, screened EV of at least 5%, a minimum 20¢ ask, and a $5 all-in cap.
The screen includes a five-point probability haircut, estimated fees, and
1¢ per share in slippage.

Each signal compares two hypothetical fill-or-kill orders at the same limit.
One uses the full screened quantity and the other uses half as many whole
shares, rounded down to the nearest share. The runner waits the specified
delay, fetches an uncached depth snapshot, and records whether it showed enough
depth for each order. Reported costs use the worst executable level in that
snapshot, plus the fee and slippage estimates. Actual execution can differ
as displayed liquidity changes or other orders reach it first.

The runner saves signals and exchange settlements to the SQLite ledger in
`data/paper_trades/nba_moneyline_shadow`. It overwrites `signals.csv` and
`summary.csv` from that ledger on each scan. The summary includes fill rate,
settled ROI, and P&L per evaluated signal, which includes observed zero fills.
Missing or stale snapshots are listed as unobserved. A smaller order below
the market minimum is marked inapplicable. The four rows represent separate
hypothetical strategies, including any signals they share.

Use Ctrl+C to stop live capture. Start the same command later to resume the
ledger. To update settlements and export results without scanning live games,
run the report command.

```powershell
python -m src.nba_moneyline_shadow --report
```

You can set the default delay with `NBA_SHADOW_LATENCY_SECONDS` in `.env`.
Use `--output` to give a different latency test its own ledger directory.

Use `--explain` to show the reason each game was skipped. The runner reports
ESPN summary errors and Polymarket order-book rate limits separately from
signals that miss the probability, price, or EV thresholds.

```powershell
python -m src.nba_moneyline_shadow --once --explain
```
