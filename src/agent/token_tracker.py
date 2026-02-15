"""Track token usage and cost for agent loop sessions.

Enforces a per-session budget to prevent runaway costs.
Supports cache-aware cost calculation for Anthropic prompt caching.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Cost per million tokens (USD)
MODEL_PRICING: dict[str, tuple[float, float]] = {
    # (input_per_mtok, output_per_mtok)
    "claude-sonnet-4-5-20250929": (3.0, 15.0),
    "claude-opus-4-6": (5.0, 25.0),
    "claude-haiku-4-5-20251001": (1.0, 5.0),
}

# Fallback pricing for unknown models
_DEFAULT_PRICING = (3.0, 15.0)

# Cache pricing multipliers (relative to base input price)
CACHE_WRITE_MULTIPLIER = 1.25  # 5-min cache writes cost 1.25x base input
CACHE_READ_MULTIPLIER = 0.10  # Cache hits cost 0.10x base input


@dataclass
class TokenEntry:
    """Single API call token record."""

    model: str
    input_tokens: int
    output_tokens: int
    cache_creation_tokens: int
    cache_read_tokens: int
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
        cache_creation_tokens: int = 0,
        cache_read_tokens: int = 0,
    ) -> None:
        """Record a single API call's token usage.

        Args:
            model: Model ID used.
            input_tokens: Input token count (non-cached).
            output_tokens: Output token count.
            turn: Turn number in the agent loop.
            cache_creation_tokens: Tokens written to cache (1.25x input price).
            cache_read_tokens: Tokens read from cache (0.10x input price).
        """
        pricing = MODEL_PRICING.get(model, _DEFAULT_PRICING)
        base_input = pricing[0]
        output_price = pricing[1]
        cost = (
            input_tokens * base_input
            + cache_creation_tokens * base_input * CACHE_WRITE_MULTIPLIER
            + cache_read_tokens * base_input * CACHE_READ_MULTIPLIER
            + output_tokens * output_price
        ) / 1_000_000
        self.entries.append(
            TokenEntry(
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cache_creation_tokens=cache_creation_tokens,
                cache_read_tokens=cache_read_tokens,
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

    @property
    def total_cache_creation_tokens(self) -> int:
        """Sum of cache creation tokens across all calls."""
        return sum(e.cache_creation_tokens for e in self.entries)

    @property
    def total_cache_read_tokens(self) -> int:
        """Sum of cache read tokens across all calls."""
        return sum(e.cache_read_tokens for e in self.entries)

    @property
    def cache_savings_usd(self) -> float:
        """How much caching saved vs. paying full input price for cache_read tokens."""
        savings = 0.0
        for e in self.entries:
            if e.cache_read_tokens > 0:
                pricing = MODEL_PRICING.get(e.model, _DEFAULT_PRICING)
                base_input = pricing[0]
                full_cost = e.cache_read_tokens * base_input / 1_000_000
                cached_cost = e.cache_read_tokens * base_input * CACHE_READ_MULTIPLIER / 1_000_000
                savings += full_cost - cached_cost
        return savings

    def summary(self) -> dict:
        """Return a summary dict for logging/persistence.

        Returns:
            Dict with total tokens, cost, turns, cache stats, and per-entry breakdown.
        """
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cache_creation_tokens": self.total_cache_creation_tokens,
            "total_cache_read_tokens": self.total_cache_read_tokens,
            "total_cost_usd": round(self.total_cost_usd, 4),
            "cache_savings_usd": round(self.cache_savings_usd, 4),
            "budget_usd": self.session_budget_usd,
            "budget_remaining_usd": round(self.budget_remaining_usd, 4),
            "turns": len(self.entries),
            "entries": [
                {
                    "model": e.model,
                    "input_tokens": e.input_tokens,
                    "output_tokens": e.output_tokens,
                    "cache_creation_tokens": e.cache_creation_tokens,
                    "cache_read_tokens": e.cache_read_tokens,
                    "cost_usd": round(e.cost_usd, 6),
                    "turn": e.turn,
                }
                for e in self.entries
            ],
        }
