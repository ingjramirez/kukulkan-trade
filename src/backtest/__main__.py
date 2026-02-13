"""CLI entry point: python -m src.backtest --months 6"""

import argparse
import asyncio

import structlog

from src.backtest.runner import BacktestRunner

structlog.configure(
    processors=[
        structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(20),
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Kukulkan — Backtest Runner")
    parser.add_argument(
        "--months",
        type=int,
        default=6,
        help="Months of history to simulate (default: 6)",
    )
    parser.add_argument(
        "--db",
        type=str,
        default="data/backtest.db",
        help="Output database path (default: data/backtest.db)",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Drop and recreate all tables before running",
    )
    parser.add_argument(
        "--use-ai",
        action="store_true",
        help="Use real Claude AI for Portfolio B (costs API tokens)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Estimate token cost without running the backtest",
    )
    parser.add_argument(
        "--ai-budget",
        type=float,
        default=1.50,
        help="Maximum USD to spend on AI API calls (default: 1.50)",
    )
    parser.add_argument(
        "--ai-strategy",
        type=str,
        default=None,
        choices=["conservative", "standard", "aggressive"],
        help="AI strategy persona — uses same directives as production (default: conservative)",
    )
    parser.add_argument(
        "--ai-prompt-override",
        type=str,
        default=None,
        help="Path to a text file with custom system prompt (overrides --ai-strategy)",
    )
    parser.add_argument(
        "--run-label",
        type=str,
        default=None,
        help="Label for this run (default: auto-detected from strategy/prompt)",
    )
    args = parser.parse_args()

    mode = "mock"
    if args.use_ai:
        mode = "AI"
    if args.dry_run:
        mode = "dry-run"

    # Load prompt override from file if provided
    prompt_override = None
    if args.ai_prompt_override:
        with open(args.ai_prompt_override) as f:
            prompt_override = f.read().strip()
        print(f"Loaded prompt override from {args.ai_prompt_override}")

    # Resolve strategy mode (--ai-prompt-override overrides --ai-strategy)
    strategy_mode = args.ai_strategy
    if not strategy_mode and not prompt_override:
        strategy_mode = "conservative"  # default matches production

    # Auto-detect run label
    run_label = args.run_label
    if run_label is None and args.use_ai:
        if strategy_mode:
            run_label = strategy_mode
        elif prompt_override and "conservative" in prompt_override.lower():
            run_label = "conservative"
        elif prompt_override and "aggressive" in prompt_override.lower():
            run_label = "aggressive"
        else:
            run_label = "standard"

    budget_str = f", budget=${args.ai_budget:.2f}" if args.use_ai else ""
    label_str = f", label={run_label}" if run_label else ""
    strategy_str = f", strategy={strategy_mode}" if strategy_mode else ""
    print(
        f"Starting backtest: {args.months} months → {args.db}"
        f" (mode={mode}{budget_str}{strategy_str}{label_str}"
        f"{', clean' if args.clean else ''})"
    )

    runner = BacktestRunner(db_path=args.db)
    summary = asyncio.run(
        runner.run(
            months=args.months,
            clean=args.clean,
            use_ai=args.use_ai,
            dry_run=args.dry_run,
            ai_budget=args.ai_budget,
            prompt_override=prompt_override,
            strategy_mode=strategy_mode,
            run_label=run_label or "default",
        )
    )

    if summary.get("dry_run"):
        print("\n=== Dry Run Estimate ===")
        print(f"  Simulation days: {summary.get('sim_days', 0)}")
        print(f"  API calls: {summary.get('estimated_api_calls', 0)}")
        print(f"  Est. tokens: {summary.get('estimated_tokens', 0):,}")
        print(f"  Est. cost: ${summary.get('estimated_cost_usd', 0):.2f}")
        print(f"  Budget: ${args.ai_budget:.2f}")
        print(f"  Note: {summary.get('note', '')}")
        return

    print("\n=== Backtest Summary ===")
    for key in ("portfolio_A", "portfolio_B"):
        data = summary.get(key, {})
        label = key.replace("portfolio_", "Portfolio ")
        print(f"\n{label}:")
        print(f"  Total Return: {data.get('total_return_pct', 0):+.2f}%")
        print(f"  Max Drawdown: {data.get('max_drawdown_pct', 0):.2f}%")
        print(f"  Final Value:  ${data.get('final_value', 0):,.2f}")
        print(f"  Snapshots:    {data.get('snapshots', 0)}")

    counts = summary.get("trade_counts", {})
    print(f"\nTotal Trades: A={counts.get('A', 0)}, B={counts.get('B', 0)}")

    # Print AI cost report if available
    ai_report = summary.get("ai_cost_report")
    if ai_report:
        print(f"\n{ai_report}")

    decisions_path = summary.get("ai_decisions_path")
    if decisions_path:
        print(f"\nDecisions saved to: {decisions_path}")


if __name__ == "__main__":
    main()
