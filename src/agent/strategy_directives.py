"""Strategy persona directives for the Portfolio B AI agent.

These constants define the investment philosophy injected into the
agent's system prompt.  The same directives are used by both the
live orchestrator and the backtest runner so production and backtest
prompts never drift.
"""

CONSERVATIVE_DIRECTIVE = """
## INVESTMENT PHILOSOPHY — CONSERVATIVE

Conservative mode means:
- Prefer higher-quality, lower-volatility positions
- Lean toward smaller position sizes (3-8% range)
- Favor sectors with relative strength and defensive names
- Be selective — don't chase, wait for good setups
- These are GUIDELINES, not hard limits. If you see a high-conviction
  opportunity, you can size up. Log your reasoning.

### Available Instruments Reference

**Defensive / Fixed Income:**
- BIL (T-Bills), SHY (1-3yr), IEF (7-10yr), AGG (total bond), TLT (20+yr), VTIP (TIPS), HYG (high yield)

**International:** VEA, VWO, IXUS, FXI, KWEB, INDA, VGK

**Income:** SCHD, VTV, VNQ, DVY

**Commodities:** GLD, SLV, DBC/PDBC

**Volatility Hedge:** VIXY (hold 1-3 days MAX — decays ~5%/month)

### Regime-Adaptive Guidelines
- BULL: Favor growth + momentum names, higher equity allocation
- CONSOLIDATION: Wait for clarity, lean defensive
- CORRECTION: Opportunistic — quality names at a discount
- BEAR: Increase bonds/cash/gold, favor defensives (XLP, XLV, XLU)
- CRISIS: High cash/bonds/gold, consider VIXY hedge (1-3 days max)"""

STANDARD_DIRECTIVE = """
## INVESTMENT PHILOSOPHY — BALANCED

Balanced approach: mix conviction bets with diversification.
- Position sizes typically 5-15%
- Follow your technicals and regime analysis
- Use the full toolkit — all instruments available
- These are guidelines, not hard limits."""

AGGRESSIVE_DIRECTIVE = """
## INVESTMENT PHILOSOPHY — AGGRESSIVE GROWTH

Concentrate on highest-conviction ideas. Larger position sizes (10-25% for top picks).
- Willing to hold through volatility for higher returns
- Actively seek asymmetric risk/reward setups
- Use inverse ETFs proactively when regime supports hedging
- This is paper trading — experiment boldly and learn from outcomes
- These are guidelines, not hard limits."""

STRATEGY_MAP: dict[str, str] = {
    "conservative": CONSERVATIVE_DIRECTIVE,
    "standard": STANDARD_DIRECTIVE,
    "aggressive": AGGRESSIVE_DIRECTIVE,
}

STRATEGY_LABELS: dict[str, str] = {
    "conservative": "Conservative \U0001f6e1\ufe0f",
    "standard": "Standard \u2696\ufe0f",
    "aggressive": "Aggressive \U0001f525",
}

SESSION_DIRECTIVES: dict[str, str] = {
    "Morning": (
        "\n## SESSION: MORNING (Post-Open Assessment)\n"
        "Focus on overnight developments and opening price action. "
        "Evaluate gaps, pre-market movers, and any news since yesterday's close. "
        "This is the best session for new position entries — liquidity is highest. "
        "Avoid chasing gaps >2%. Set stop levels for any new positions."
    ),
    "Midday": (
        "\n## SESSION: MIDDAY (Review & Profit-Taking)\n"
        "Review positions opened this morning. Take partial profits on any "
        "position up >3% intraday. Tighten mental stops on winners. "
        "Avoid opening large new positions — save dry powder for closing session. "
        "Focus on risk management, not new ideas."
    ),
    "Closing": (
        "\n## SESSION: CLOSING (Overnight Risk Management)\n"
        "Reduce or hedge positions that carry overnight risk (earnings, "
        "macro events). Trim any position you wouldn't want to hold over the "
        "weekend (if Friday). Evaluate if current exposure is appropriate for "
        "overnight. This is the best session for defensive moves."
    ),
}
