# Polymarket US wager button

Open the live scanner, choose **Review wager** beside a positive-EV pick, and enter the number of contracts. The ticket checks the exact game, full-game market, side, and line on Polymarket US. It shows the US price limit, estimated fees and maximum cost, screened US EV, and the US settlement terms. International prices remain on the scanner and are separate from this ticket.

**Open on Polymarket US** sits beside each wager and opens the matching US game in a new tab with the exact market and outcome selected. You can also open the link from the review dialog. The link uses the verified listing and is available independently of the order-book preview. If the exact US market or side is unavailable, the dialog explains why it cannot open it. Complete the wager on Polymarket US if you choose to proceed.

On a regular installation the ticket shows a quote. Direct orders require an eligible Polymarket US account and API credentials, plus configuration for a private single-user installation. Configure these environment variables on the server:

| Name | Value |
|---|---|
| `POLYMARKET_KEY_ID` and `POLYMARKET_SECRET_KEY` | The Polymarket US API key ID and signing secret |
| `PARLAY_GUARD_ORDER_TOKEN` | A unique random token with at least 32 characters. Enter this token in the wager dialog |
| `PARLAY_GUARD_ORDER_ORIGIN` | Exact origin of the private dashboard, such as `http://127.0.0.1:8000` or its HTTPS origin |
| `PARLAY_GUARD_ORDER_DB` | Absolute path to a writable SQLite file on persistent storage |
| `PARLAY_GUARD_US_ORDERS` | `1` to enable authenticated order requests |

Keep credentials and the access token in local or deployment secrets. For hosted use, use an HTTPS private instance and durable storage for the ticket database. Start the app with `python -m uvicorn src.api:app --host 127.0.0.1 --port 8000` for local access.

**Confirm real-money order** sends one fill-or-kill buy limit order. The exchange either fills the full quantity at the limit or better or cancels it. The server checks the live pick, US order book, price and screened EV again on confirmation. A used ticket cannot be submitted again. If the exchange response is interrupted, check order activity in Polymarket US before requesting another order.

The signed request uses the [Polymarket US Python SDK](https://github.com/Polymarket/polymarket-us-python) and standard [order pricing conventions](https://docs.polymarket.us/api-reference/orders/overview). The fee estimate uses the published standard taker fee with an upward rounding allowance. Actual exchange charges may differ by configured coefficients and fills.
