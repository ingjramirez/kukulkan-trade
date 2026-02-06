"""CLI entry point: python -m src.backtest --months 6"""

import argparse
import asyncio
import sys

import structlog

from src.backtest.runner import BacktestRunner

structlog.configure(
    processors=[
        structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(20),
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Atlas Trading Bot — Backtest Runner")
    parser.add_argument("--months", type=int, default=6, help="Months of history to simulate (default: 6)")
    parser.add_argument("--db", type=str, default="data/backtest.db", help="Output database path (default: data/backtest.db)")
    parser.add_argument("--clean", action="store_true", help="Drop and recreate all tables before running")
    args = parser.parse_args()

    print(f"Starting backtest: {args.months} months → {args.db}{' (clean)' if args.clean else ''}")
    runner = BacktestRunner(db_path=args.db)
    summary = asyncio.run(runner.run(months=args.months, clean=args.clean))

    print("\n=== Backtest Summary ===")
    for key in ("portfolio_A", "portfolio_B", "portfolio_C"):
        data = summary.get(key, {})
        label = key.replace("portfolio_", "Portfolio ")
        print(f"\n{label}:")
        print(f"  Total Return: {data.get('total_return_pct', 0):+.2f}%")
        print(f"  Max Drawdown: {data.get('max_drawdown_pct', 0):.2f}%")
        print(f"  Final Value:  ${data.get('final_value', 33_333):,.2f}")
        print(f"  Snapshots:    {data.get('snapshots', 0)}")

    counts = summary.get("trade_counts", {})
    print(f"\nTotal Trades: A={counts.get('A', 0)}, B={counts.get('B', 0)}, C={counts.get('C', 0)}")


if __name__ == "__main__":
    main()
