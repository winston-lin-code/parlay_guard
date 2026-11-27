# Women’s college basketball live orders

Start the WCBB Anytime strategy in real-money mode from the repository root.
The runner reads the Polymarket US credentials from the project `.env` and
authenticates with an account balance read before starting the scanner.

```powershell
python -m src.run_research --live-wcbb
```

The strategy looks for its first qualifying in-game moneyline with win
probability above 60%, screened EV of at least 5%, and a best ask of at least
20¢. It attempts one fill-or-kill limit order per game with a $5 all-in cap.

The WCBB Anytime tab shows order IDs, exchange states, and confirmed fill
quantities alongside historical paper results. Keep
`data/live_orders/wcbb_anytime.sqlite3` between runs to retain the
one-attempt-per-game limit. Press Ctrl+C to stop automatic orders.
[WCBB Anytime dashboard](http://127.0.0.1:8000/inplay?phase=wcbb)
