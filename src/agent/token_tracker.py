"""Track token usage and cost for agent loop sessions.

Enforces a per-session budget to prevent runaway costs.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Cost per million tokens (USD)
MODEL_PRICING: dict[str, tuple[float, float]] = {
    # (input_per_mtok, output_per_mtok)
    "claude-sonnet-4-5-20250929": (3.0, 15.0),
    "claude-opus-4-6": (15.0, 75.0),
    "claude-haiku-4-5-20251001": (0.80, 4.0),
}

# Fallback pricing for unknown models
_DEFAULT_PRICING = (3.0, 15.0)


@dataclass
class TokenEntry:
    """Single API call token record."""

    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    turn: int


@dataclass
class TokenTracker:
    """Tracks token usage and cost across an agent session.

    Args:
        session_budget_usd: Maximum spend for this session.
    """

    session_budget_usd: float = 0.50
    entries: list[TokenEntry] = field(default_factory=list)

    def record(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        turn: int,
    ) -> None:
        """Record a single API call's token usage.

        Args:
            model: Model ID used.
            input_tokens: Input token count.
            output_tokens: Output token count.
            turn: Turn number in the agent loop.
        """
        pricing = MODEL_PRICING.get(model, _DEFAULT_PRICING)
        cost = (input_tokens * pricing[0] + output_tokens * pricing[1]) / 1_000_000
        self.entries.append(
            TokenEntry(
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost,
                turn=turn,
            )
        )

    @property
    def total_cost_usd(self) -> float:
        """Total cost across all entries."""
        return sum(e.cost_usd for e in self.entries)

    @property
    def budget_remaining_usd(self) -> float:
        """Remaining budget."""
        return self.session_budget_usd - self.total_cost_usd

    @property
    def budget_exceeded(self) -> bool:
        """Whether the session budget has been exceeded."""
        return self.total_cost_usd >= self.session_budget_usd

    @property
    def total_input_tokens(self) -> int:
        """Sum of input tokens across all calls."""
        return sum(e.input_tokens for e in self.entries)

    @property
    def total_output_tokens(self) -> int:
        """Sum of output tokens across all calls."""
        return sum(e.output_tokens for e in self.entries)

    def summary(self) -> dict:
        """Return a summary dict for logging/persistence.

        Returns:
            Dict with total tokens, cost, turns, and per-entry breakdown.
        """
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost_usd": round(self.total_cost_usd, 4),
            "budget_usd": self.session_budget_usd,
            "budget_remaining_usd": round(self.budget_remaining_usd, 4),
            "turns": len(self.entries),
            "entries": [
                {
                    "model": e.model,
                    "input_tokens": e.input_tokens,
                    "output_tokens": e.output_tokens,
                    "cost_usd": round(e.cost_usd, 6),
                    "turn": e.turn,
                }
                for e in self.entries
            ],
        }
