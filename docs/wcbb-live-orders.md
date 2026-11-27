# Women’s college basketball live orders

Start the WCBB Anytime strategy in real-money mode from the repository root.
The runner reads the Polymarket US credentials from the project `.env` and
authenticates with an account balance read before starting the scanner.

```powershell
python -m src.run_research --live-wcbb
```

The strategy looks for its first qualifying in-game moneyline with win
probability above 60%, screened EV of at least 5%, and a best ask of at least
20¢. It uses a fill-or-kill limit order with a $5 all-in cap per game.

Live mode subscribes to full Polymarket US order books over one WebSocket.
It includes matched live games and listed moneylines starting within five
minutes. If a stream snapshot is missing or stale, the scanner can use up to
four paced REST book requests per scan and honors the exchange cooldown.

If Polymarket confirms the first order expired with zero fills, the runner
refreshes the score and order book and allows one retry that passes the same
screen. A request with unknown status stays reserved while Polymarket status
is checked by order ID when available.

The WCBB Anytime tab shows order IDs, exchange states, and confirmed fill
quantities alongside historical paper results. Keep
`data/live_orders/wcbb_anytime.sqlite3` between runs to preserve both attempts
and the per-game cap. Press Ctrl+C to stop automatic orders.
[WCBB Anytime dashboard](http://127.0.0.1:8000/inplay?phase=wcbb)
