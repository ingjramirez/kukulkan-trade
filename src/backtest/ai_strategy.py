"""AI backtest strategy with budget tracking and decision logging.

Wraps AIAutonomyStrategy + ClaudeAgent for backtesting with real API calls,
cost tracking, budget enforcement, and JSON decision persistence.
"""

import json
import os
from datetime import date, datetime

import pandas as pd
import structlog

from src.agent.claude_agent import ClaudeAgent, build_system_prompt
from src.agent.strategy_directives import STRATEGY_MAP
from src.storage.models import TradeSchema
from src.strategies.portfolio_b import AIAutonomyStrategy

log = structlog.get_logger()

# Claude Sonnet 4.5 pricing (per million tokens)
INPUT_COST_PER_M = 3.0
OUTPUT_COST_PER_M = 15.0


class BudgetExhaustedError(Exception):
    """Raised when the AI budget has been exceeded."""


class AIBacktestStrategy:
    """Portfolio B AI strategy with budget tracking for backtesting.

    Tracks input/output tokens and cost per call, enforces a budget cap,
    and logs every decision to JSON for later analysis.

    Args:
        budget_usd: Maximum spend before halting API calls.
        run_label: Label for this run (e.g. "standard", "conservative").
        prompt_override: Custom system prompt text (overrides default).
        strategy_mode: Strategy persona — uses same directives as production.
        decisions_dir: Directory for JSON decision logs.
        agent: Optional pre-configured ClaudeAgent.
    """

    def __init__(
        self,
        budget_usd: float = 1.50,
        run_label: str = "default",
        prompt_override: str | None = None,
        strategy_mode: str | None = None,
        decisions_dir: str = "data/backtest_decisions",
        agent: ClaudeAgent | None = None,
    ) -> None:
        self._budget_usd = budget_usd
        self._run_label = run_label
        self._decisions_dir = decisions_dir

        self._agent = agent or ClaudeAgent()
        self._strategy = AIAutonomyStrategy(agent=self._agent)

        # Resolve effective system prompt:
        # 1. Explicit prompt_override text takes priority (legacy / custom)
        # 2. strategy_mode uses the same directives as production
        # 3. Default: conservative (matches production default)
        if prompt_override:
            self._prompt_override = prompt_override
        elif strategy_mode and strategy_mode in STRATEGY_MAP:
            self._prompt_override = build_system_prompt(strategy_mode=strategy_mode)
        else:
            self._prompt_override = None

        # Cost tracking
        self._total_input_tokens = 0
        self._total_output_tokens = 0
        self._total_cost_usd = 0.0
        self._api_calls = 0
        self._decisions: list[dict] = []

    @property
    def total_cost_usd(self) -> float:
        return self._total_cost_usd

    @property
    def total_tokens(self) -> int:
        return self._total_input_tokens + self._total_output_tokens

    @property
    def api_calls(self) -> int:
        return self._api_calls

    @property
    def budget_remaining(self) -> float:
        return max(0, self._budget_usd - self._total_cost_usd)

    @property
    def budget_exhausted(self) -> bool:
        return self._total_cost_usd >= self._budget_usd

    def _track_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Record token usage and return cost for this call."""
        cost = (input_tokens * INPUT_COST_PER_M / 1_000_000) + (output_tokens * OUTPUT_COST_PER_M / 1_000_000)
        self._total_input_tokens += input_tokens
        self._total_output_tokens += output_tokens
        self._total_cost_usd += cost
        self._api_calls += 1
        return cost

    def generate_trades(
        self,
        closes: pd.DataFrame,
        volumes: pd.DataFrame,
        positions: list[dict],
        cash: float,
        total_value: float,
        current_positions: dict[str, float],
        recent_trades: list[dict],
        sim_date: date,
    ) -> list[TradeSchema]:
        """Call Claude and return validated trades with budget enforcement.

        Args:
            closes: Historical closes up to sim_date.
            volumes: Historical volumes.
            positions: Current positions as list of dicts.
            cash: Available cash.
            total_value: Total portfolio value.
            current_positions: Dict of ticker -> shares held.
            recent_trades: Recent trade history.
            sim_date: Current simulation date.

        Returns:
            List of validated TradeSchema objects. Empty if budget exhausted.
        """
        if self.budget_exhausted:
            log.info(
                "ai_backtest_budget_exhausted",
                date=str(sim_date),
                total_cost=f"${self._total_cost_usd:.4f}",
            )
            return []

        context = self._strategy.prepare_context(
            closes=closes,
            volumes=volumes,
            positions=positions,
            cash=cash,
            total_value=total_value,
            recent_trades=recent_trades,
        )

        # Override the analysis date to the simulation date
        context["analysis_date"] = sim_date

        # Apply prompt override if provided
        if self._prompt_override:
            context["system_prompt"] = self._prompt_override

        try:
            response = self._strategy._agent.analyze(**context)
        except Exception as e:
            log.error(
                "ai_backtest_api_error",
                date=str(sim_date),
                error=str(e),
            )
            self._log_decision(sim_date, {}, [], error=str(e))
            return []

        # Track cost from response metadata
        tokens_used = response.get("_tokens_used", 0)
        # Estimate split: ~75% input, ~25% output
        input_tokens = int(tokens_used * 0.75)
        output_tokens = tokens_used - input_tokens
        call_cost = self._track_cost(input_tokens, output_tokens)

        # Convert to trades
        trades = self._strategy.agent_response_to_trades(
            response=response,
            total_value=total_value,
            current_positions=current_positions,
            latest_prices=closes.iloc[-1],
        )

        self._log_decision(sim_date, response, trades, cost=call_cost)

        log.info(
            "ai_backtest_decision",
            date=str(sim_date),
            trades=len(trades),
            tokens=tokens_used,
            cost=f"${call_cost:.4f}",
            total_cost=f"${self._total_cost_usd:.4f}",
            budget_remaining=f"${self.budget_remaining:.4f}",
        )

        return trades

    def _log_decision(
        self,
        sim_date: date,
        response: dict,
        trades: list[TradeSchema],
        cost: float = 0.0,
        error: str | None = None,
    ) -> None:
        """Record a decision for later JSON export."""
        entry = {
            "date": sim_date.isoformat(),
            "timestamp": datetime.utcnow().isoformat(),
            "run_label": self._run_label,
            "regime_assessment": response.get("regime_assessment", ""),
            "reasoning": response.get("reasoning", ""),
            "risk_notes": response.get("risk_notes", ""),
            "trades": [
                {
                    "ticker": t.ticker,
                    "side": t.side.value,
                    "shares": t.shares,
                    "price": t.price,
                    "reason": t.reason or "",
                }
                for t in trades
            ],
            "tokens_used": response.get("_tokens_used", 0),
            "model": response.get("_model", ""),
            "cost_usd": round(cost, 6),
            "cumulative_cost_usd": round(self._total_cost_usd, 6),
        }
        if error:
            entry["error"] = error
        self._decisions.append(entry)

    def save_decisions(self) -> str | None:
        """Write all decisions to a JSON file.

        Returns:
            Path to the saved file, or None if no decisions.
        """
        if not self._decisions:
            return None

        os.makedirs(self._decisions_dir, exist_ok=True)
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = f"{self._run_label}_{ts}.json"
        path = os.path.join(self._decisions_dir, filename)

        output = {
            "run_label": self._run_label,
            "budget_usd": self._budget_usd,
            "total_cost_usd": round(self._total_cost_usd, 6),
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
            "api_calls": self._api_calls,
            "decisions": self._decisions,
        }

        with open(path, "w") as f:
            json.dump(output, f, indent=2)

        log.info("ai_backtest_decisions_saved", path=path, count=len(self._decisions))
        return path

    def get_cost_report(self) -> str:
        """Generate a human-readable cost summary.

        Returns:
            Formatted cost report string.
        """
        lines = [
            f"=== AI Backtest Cost Report ({self._run_label}) ===",
            f"  API Calls:      {self._api_calls}",
            f"  Input Tokens:   {self._total_input_tokens:,}",
            f"  Output Tokens:  {self._total_output_tokens:,}",
            f"  Total Tokens:   {self.total_tokens:,}",
            f"  Total Cost:     ${self._total_cost_usd:.4f}",
            f"  Budget:         ${self._budget_usd:.2f}",
            f"  Budget Used:    {(self._total_cost_usd / self._budget_usd * 100):.1f}%"
            if self._budget_usd > 0
            else "  Budget Used:    N/A",
        ]
        return "\n".join(lines)
