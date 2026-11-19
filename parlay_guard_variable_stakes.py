"""Backtest variable wager sizing on Parlay Guard's Polymarket bet candidates.

INPUT: the *_bets.csv output from parlay_guard_polymarket_full_backtest.py.

Example:
    python parlay_guard_variable_stakes.py \
      --input parlay_guard_polymarket_bets.csv \
      --bankroll 1000 \
      --flat-stake 10 \
      --output-prefix parlay_guard_variable

By default, compare wager sizing on the SAME value-filtered bets
(strategies containing 'EV > 5%') to make sizing comparisons fair.
Each day uses the SAME starting bankroll for all bets placed that day;
settlement profits/losses are applied at the end of the day. Total
same-day stakes are capped as a percentage of that day's bankroll.

IMPORTANT: Input prices are historical sampled mid/indicative prices,
not guaranteed executable asks. The +1c/+2c input scenarios are
execution-cost sensitivities, not real order-book execution records.
There is no claim of live-tradable profitability or model calibration.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


SIZING_METHODS = (
    "flat",
    "ev_tiered",
    "ev_scaled_2pct",
    "quarter_kelly_2pct",
    "quarter_kelly_5pct",
    "half_kelly_5pct",
)


def kelly_fraction(probability: float, price: float) -> float:
    """Fraction of bankroll maximizing expected log growth for binary shares."""
    if not (0 < price < 1 and 0 <= probability <= 1):
        return 0.0
    return max(0.0, (probability - price) / (1.0 - price))


def desired_wager(method: str, bankroll: float, prob: float,
                  price: float, flat_stake: float) -> float:
    """Size the wager using only information available before the game."""
    edge_roi = prob / price - 1.0
    if edge_roi <= 0 or bankroll <= 0:
        return 0.0
    if method == "flat":
        return min(flat_stake, bankroll)
    if method == "ev_tiered":
        # Increasing absolute dollar stake for larger expected ROI.
        if edge_roi < 0.10:
            stake = 5.0
        elif edge_roi < 0.20:
            stake = 10.0
        elif edge_roi < 0.35:
            stake = 15.0
        else:
            stake = 20.0
        return min(stake, 0.02 * bankroll)
    if method == "ev_scaled_2pct":
        # 5% projected ROI => 0.25% bankroll; 20% => 1%; 40% => 2%.
        return min(0.02, 0.05 * edge_roi) * bankroll
    kelly = kelly_fraction(prob, price)
    if method == "quarter_kelly_2pct":
        return min(0.25 * kelly, 0.02) * bankroll
    if method == "quarter_kelly_5pct":
        return min(0.25 * kelly, 0.05) * bankroll
    if method == "half_kelly_5pct":
        return min(0.50 * kelly, 0.05) * bankroll
    raise ValueError(f"Unknown sizing method: {method}")


def read_candidates(path: Path, only_value: bool) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        records = list(csv.DictReader(f))
    if not records:
        raise ValueError(f"No bet candidates found in {path}")
    needed = {
        "scenario_buy_premium_cents", "strategy", "game_id", "date",
        "selected", "model_p_used", "assumed_buy_price", "won",
    }
    missing = needed - set(records[0])
    if missing:
        raise ValueError(f"Missing columns in input CSV: {sorted(missing)}")
    out = []
    seen = set()
    for row in records:
        if only_value and "EV > 5%" not in row["strategy"]:
            continue
        key = (row["scenario_buy_premium_cents"], row["strategy"], row["game_id"])
        if key in seen:
            raise ValueError(f"Duplicate bet candidate: {key}")
        seen.add(key)
        price = float(row["assumed_buy_price"])
        prob = float(row["model_p_used"])
        if not (0 < price < 1 and 0 <= prob <= 1):
            continue
        out.append({
            "premium_cents": int(row["scenario_buy_premium_cents"]),
            "strategy": row["strategy"],
            "game_id": row["game_id"],
            "date": row["date"],
            "selected": row["selected"],
            "probability": prob,
            "price": price,
            "expected_roi": prob / price - 1,
            "won": int(row["won"]),
        })
    return out


def simulate_sizing(candidates: list[dict], bankroll: float, flat_stake: float,
                    max_day_pct: float) -> tuple[list[dict], list[dict], list[dict]]:
    groups = defaultdict(list)
    for row in candidates:
        groups[(row["premium_cents"], row["strategy"])].append(row)

    bet_rows = []
    summaries = []
    daily_rows = []
    for (premium, strategy), group in sorted(groups.items()):
        dates = defaultdict(list)
        for r in group:
            dates[r["date"]].append(r)
        for method in SIZING_METHODS:
            equity = bankroll
            peak = bankroll
            max_drawdown = 0.0
            max_drawdown_pct = 0.0
            bets = 0
            wins = 0
            total_staked = 0.0
            total_profit = 0.0
            max_wager = 0.0
            for date, day_bets in sorted(dates.items()):
                if equity <= 0:
                    break
                start_equity = equity
                ordered = sorted(day_bets, key=lambda r: r["game_id"])
                proposed = [desired_wager(method, start_equity, r["probability"],
                                          r["price"], flat_stake) for r in ordered]
                # Avoid treating winnings from earlier same-day games as
                # already available for bets placed before those games end.
                day_limit = max_day_pct / 100.0 * start_equity
                total_proposed = sum(proposed)
                scale = min(1.0, day_limit / total_proposed) if total_proposed else 1.0
                day_profit = 0.0
                day_staked = 0.0
                day_count = 0
                for r, proposed_stake in zip(ordered, proposed):
                    stake = round(proposed_stake * scale, 2)
                    if stake < 0.01:
                        continue
                    shares = stake / r["price"]
                    profit = shares - stake if r["won"] else -stake
                    day_profit += profit
                    day_staked += stake
                    day_count += 1
                    bets += 1
                    wins += r["won"]
                    total_staked += stake
                    total_profit += profit
                    max_wager = max(max_wager, stake)
                    bet_rows.append({
                        "premium_cents": premium,
                        "model_strategy": strategy,
                        "sizing_method": method,
                        "date": date,
                        "game_id": r["game_id"],
                        "selected": r["selected"],
                        "model_probability": round(r["probability"], 6),
                        "assumed_buy_price": round(r["price"], 6),
                        "expected_roi_pct": round(100 * r["expected_roi"], 3),
                        "start_of_day_bankroll": round(start_equity, 2),
                        "wager": stake,
                        "shares": round(shares, 6),
                        "won": r["won"],
                        "net_profit": round(profit, 4),
                    })
                equity += day_profit
                peak = max(peak, equity)
                drawdown = peak - equity
                max_drawdown = max(max_drawdown, drawdown)
                max_drawdown_pct = max(max_drawdown_pct, 100 * drawdown / peak if peak else 0)
                daily_rows.append({
                    "premium_cents": premium,
                    "model_strategy": strategy,
                    "sizing_method": method,
                    "date": date,
                    "bets": day_count,
                    "total_wagered": round(day_staked, 2),
                    "daily_profit": round(day_profit, 2),
                    "ending_bankroll": round(equity, 2),
                })
            summaries.append({
                "premium_cents": premium,
                "model_strategy": strategy,
                "sizing_method": method,
                "bets": bets,
                "wins": wins,
                "win_rate_pct": round(100 * wins / bets, 2) if bets else None,
                "starting_bankroll": round(bankroll, 2),
                "ending_bankroll": round(equity, 2),
                "net_profit": round(total_profit, 2),
                "total_staked": round(total_staked, 2),
                "roi_on_stakes_pct": round(100 * total_profit / total_staked, 2) if total_staked else None,
                "bankroll_return_pct": round(100 * (equity / bankroll - 1), 2),
                "max_drawdown_usd": round(max_drawdown, 2),
                "max_drawdown_pct": round(max_drawdown_pct, 2),
                "average_wager": round(total_staked / bets, 2) if bets else None,
                "largest_wager": round(max_wager, 2),
            })
    return bet_rows, summaries, daily_rows


def write_csv(path: Path, records: list[dict]) -> None:
    if not records:
        raise ValueError(f"No data to write for {path}")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("parlay_guard_polymarket_bets.csv"))
    parser.add_argument("--bankroll", type=float, default=1000.0)
    parser.add_argument("--flat-stake", type=float, default=10.0)
    parser.add_argument("--max-day-pct", type=float, default=20.0,
                        help="Max total stakes placed in one day as %% of start-of-day bankroll")
    parser.add_argument("--all-strategies", action="store_true",
                        help="Also size non-value-filtered bets; Kelly methods skip negative-EV bets")
    parser.add_argument("--output-prefix", default="parlay_guard_variable")
    args = parser.parse_args()
    if args.bankroll <= 0 or args.flat_stake <= 0 or not 0 < args.max_day_pct <= 100:
        parser.error("bankroll, flat-stake must be positive; max-day-pct must be 0-100")
    candidates = read_candidates(args.input, only_value=not args.all_strategies)
    if not candidates:
        parser.error("No qualifying bets. Check the input and the --all-strategies flag.")
    bet_rows, summaries, daily_rows = simulate_sizing(
        candidates, args.bankroll, args.flat_stake, args.max_day_pct)
    prefix = Path(args.output_prefix)
    write_csv(Path(str(prefix) + "_bets.csv"), bet_rows)
    write_csv(Path(str(prefix) + "_summary.csv"), summaries)
    write_csv(Path(str(prefix) + "_daily_equity.csv"), daily_rows)
    print(f"Loaded {len(candidates)} value-filtered bet candidates")
    print(f"Bankroll: ${args.bankroll:,.2f}; flat stake: ${args.flat_stake:.2f}; "
          f"daily exposure cap: {args.max_day_pct:g}%")
    print(f"Wrote: {prefix}_bets.csv, {prefix}_summary.csv, {prefix}_daily_equity.csv")
    print("\nConsensus results at sampled price + 1 cent:")
    subset = [r for r in summaries if r["premium_cents"] == 1
              and r["model_strategy"].startswith("Consensus:")]
    for r in subset:
        print(f"  {r['sizing_method']:24s} {r['bets']:4d} bets | "
              f"profit ${r['net_profit']:+9.2f} | "
              f"bankroll ${r['ending_bankroll']:9.2f} | "
              f"max DD {r['max_drawdown_pct']:6.2f}%")


if __name__ == "__main__":
    main()
