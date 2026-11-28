# NBA moneyline paper study

Run the study from the Parlay Guard project folder while regular-season NBA
games are in progress. The runner uses public ESPN scores and forecasts and
Polymarket US order-book depth. It subscribes to the verified live NBA
moneylines with the Polymarket US credentials in the project `.env`.
With a configured stream it allows up to two paced REST fallbacks per scan for
initial quotes and two for delayed checks. Without usable credentials at
startup it uses paced REST.

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
delay, then allows up to 750 ms for a new streamed snapshot received at or
after the delay target. If none arrives, it uses an uncached REST snapshot
when the fallback budget and rate limit permit. It records the source and
actual delay and checks whether the snapshot showed enough depth for each
order. Reported costs use the worst executable level in that
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

## Live NBA halftime with quarter Kelly

Run this command from the project folder to check the Polymarket US credentials
in `.env` and see the available USD buying power and per-game cap.

```powershell
python -m src.nba_halftime_kelly --check-account
```

Start real-money automatic orders with the live flag.

```powershell
python -m src.nba_halftime_kelly --live --explain
```

This runner uses the same NBA halftime window, ESPN probability threshold,
20¢ minimum ask, 5% screened EV floor, and five-point haircut as the paper
study. It sizes whole shares using the account's available USD buying power.
Quarter Kelly uses the all-in cost per share, including estimated fees and a
1¢ cushion. The all-in wager is capped at 2% of buying power and $5 per game.
The fill-or-kill order uses the quoted limit on the book. Use Ctrl+C to stop
scanning. The order ledger at `data/live_orders/nba_halftime_kelly.sqlite3`
deduplicates games across restarts and permits one retry after the exchange
confirms a zero-fill expiry and a fresh quote still qualifies.
