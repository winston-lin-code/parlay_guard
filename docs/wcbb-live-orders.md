# Women’s college basketball live orders

## All-feature first entry

Build the model using the 2019–2026 research inputs and refresh the current
season's pregame context. Current-season games can update Elo, efficiency and
opponent strength only after the 24-hour results embargo. The fitted
coefficients use the historical seasons. Both teams must have current Division I
membership. The model uses verified one-minute checkpoints in regulation.

```powershell
python -m src.wcbb_history_download --data .\data\wcbb-history --refresh-context
python -m src.train_wcbb_anytime --data .\data\wcbb-history --output .\data\wcbb_all_features.json
python -m src.run_research --live-wcbb --wcbb-model .\data\wcbb_all_features.json
```

The last command enables real orders. Omit `--live-wcbb` to log paper entries
with the same model in its own paper ledger. The artifact path can also come
from `WCBB_ANYTIME_MODEL` in the project `.env`. An invalid artifact stops
startup. Refresh the context and rebuild the artifact when new season data
arrive, then restart the dashboard to load it.

Take the first verified moneyline per game with at least 60% model win
probability, screened EV above 5%, and a best ask of at least 20¢. The screen
uses a five-point probability haircut, estimated fees and 1¢ per share of
slippage. Sizing uses the $5 all-in cap and current executable depth.

## ESPN probability mode

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

In both modes, if Polymarket confirms the first order expired with zero fills, the runner
refreshes the score and order book and allows one retry that passes the same
screen. A request with unknown status stays reserved while Polymarket status
is checked by order ID when available.

The WCBB Anytime tab shows order IDs, exchange states, and confirmed fill
quantities alongside historical paper results. Keep
`data/live_orders/wcbb_anytime.sqlite3` between runs to preserve both attempts
and the per-game cap. Press Ctrl+C to stop automatic orders.
[WCBB Anytime dashboard](http://127.0.0.1:8000/inplay?phase=wcbb)
