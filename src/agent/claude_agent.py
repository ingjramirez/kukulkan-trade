"""Claude AI agent for trade analysis and Portfolio C decision-making.

Uses Anthropic Claude API (Sonnet 4.5) to:
- Analyze market conditions and news sentiment
- Generate trade proposals for Portfolio C
- Provide reasoning for the daily brief
"""

import json
from datetime import date

import anthropic
import structlog

from config.settings import settings
from config.strategies import PORTFOLIO_C

log = structlog.get_logger()

SYSTEM_PROMPT = """You are Atlas, an AI portfolio manager for an educational paper trading bot.
You manage Portfolio C ($33,333 virtual allocation) with full autonomy over a universe of ~55 tickers
including sector ETFs, thematic ETFs, inverse ETFs, commodities, individual stocks, and a Bitcoin proxy.

Your goals:
1. Maximize risk-adjusted returns over the medium term (weeks to months)
2. Demonstrate thoughtful portfolio construction with clear reasoning
3. Consider macro regime, sector rotation, momentum, valuation, and sentiment
4. Manage risk through diversification and position sizing

Constraints:
- Maximum 10 positions at any time
- No single position > 30% of portfolio value
- You must output valid JSON in the specified format

Be contrarian when the data supports it. Avoid herding into crowded trades.
Think in terms of risk/reward asymmetry, not just direction."""

ANALYSIS_PROMPT_TEMPLATE = """## Current Date: {date}

## Portfolio State
Cash: ${cash:,.2f}
Total Value: ${total_value:,.2f}
Current Positions:
{positions_text}

## Market Data (Last 5 Days Close Prices)
{price_table}

## Technical Indicators (Latest)
{indicators_table}

## Macro Context
{macro_context}

## Recent Portfolio Trades
{recent_trades}

---

Analyze the current market environment and your portfolio. Then decide on trades.

Respond ONLY with valid JSON in this exact format:
{{
  "regime_assessment": "Your 1-2 sentence macro regime read",
  "reasoning": "Your 3-5 sentence analysis of what trades to make and why",
  "trades": [
    {{
      "ticker": "XLK",
      "side": "BUY",
      "weight": 0.15,
      "reason": "brief reason for this specific trade"
    }}
  ],
  "risk_notes": "Any risk concerns or hedging rationale"
}}

Rules for the trades array:
- "side" must be "BUY" or "SELL"
- "weight" is target portfolio weight (0.0 to 0.30). Use 0.0 to fully exit a position.
- Only include tickers you want to change. Omit tickers you want to hold unchanged.
- If no trades needed, return an empty trades array.
- Ticker must be from the universe provided in the price table."""


def build_positions_text(positions: list[dict]) -> str:
    """Format current positions for the prompt.

    Args:
        positions: List of dicts with ticker, shares, avg_price, market_value.

    Returns:
        Formatted text block.
    """
    if not positions:
        return "  (no positions — fully in cash)"

    lines = []
    for p in positions:
        lines.append(
            f"  {p['ticker']}: {p['shares']:.0f} shares @ ${p['avg_price']:.2f} avg "
            f"(value: ${p.get('market_value', 0):,.2f})"
        )
    return "\n".join(lines)


def build_price_table(prices: dict[str, list[float]], tickers: list[str]) -> str:
    """Format recent close prices as a text table.

    Args:
        prices: Dict of ticker -> list of last 5 close prices.
        tickers: Ordered list of tickers to include.

    Returns:
        Formatted text table.
    """
    lines = [f"{'Ticker':<8} {'D-4':>8} {'D-3':>8} {'D-2':>8} {'D-1':>8} {'Today':>8}"]
    lines.append("-" * 50)
    for t in tickers:
        if t in prices and len(prices[t]) >= 5:
            vals = prices[t][-5:]
            lines.append(
                f"{t:<8} {vals[0]:>8.2f} {vals[1]:>8.2f} {vals[2]:>8.2f} "
                f"{vals[3]:>8.2f} {vals[4]:>8.2f}"
            )
    return "\n".join(lines)


def build_indicators_table(indicators: dict[str, dict]) -> str:
    """Format technical indicators as a text table.

    Args:
        indicators: Dict of ticker -> dict with rsi_14, macd, sma_20, sma_50.

    Returns:
        Formatted text table.
    """
    lines = [f"{'Ticker':<8} {'RSI':>6} {'MACD':>8} {'SMA20':>8} {'SMA50':>8}"]
    lines.append("-" * 44)
    for t, ind in indicators.items():
        rsi = f"{ind.get('rsi_14', 0):.1f}" if ind.get("rsi_14") is not None else "  N/A"
        macd = f"{ind.get('macd', 0):.2f}" if ind.get("macd") is not None else "    N/A"
        sma20 = f"{ind.get('sma_20', 0):.2f}" if ind.get("sma_20") is not None else "    N/A"
        sma50 = f"{ind.get('sma_50', 0):.2f}" if ind.get("sma_50") is not None else "    N/A"
        lines.append(f"{t:<8} {rsi:>6} {macd:>8} {sma20:>8} {sma50:>8}")
    return "\n".join(lines)


def build_macro_context(
    regime: str | None = None,
    yield_curve: float | None = None,
    vix: float | None = None,
) -> str:
    """Format macro indicators for the prompt.

    Args:
        regime: Current regime from Portfolio B's detector.
        yield_curve: 10Y-2Y spread.
        vix: Current VIX.

    Returns:
        Formatted text block.
    """
    lines = []
    if regime:
        lines.append(f"- Regime (from Portfolio B detector): {regime}")
    if yield_curve is not None:
        curve_status = "INVERTED" if yield_curve < 0 else "normal"
        lines.append(f"- Yield Curve (10Y-2Y): {yield_curve:+.2f}% ({curve_status})")
    if vix is not None:
        vix_status = "HIGH FEAR" if vix > 30 else "elevated" if vix > 20 else "low"
        lines.append(f"- VIX: {vix:.1f} ({vix_status})")
    return "\n".join(lines) if lines else "  (no macro data available)"


def build_recent_trades_text(trades: list[dict]) -> str:
    """Format recent trades for context.

    Args:
        trades: List of dicts with ticker, side, shares, price, reason.

    Returns:
        Formatted text block.
    """
    if not trades:
        return "  (no recent trades)"

    lines = []
    for t in trades[-5:]:  # last 5 trades
        lines.append(
            f"  {t['side']} {t['shares']:.0f}x {t['ticker']} @ ${t['price']:.2f} — {t.get('reason', '')}"
        )
    return "\n".join(lines)


class ClaudeAgent:
    """Claude-powered trading agent for Portfolio C."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = PORTFOLIO_C.model,
    ) -> None:
        self._api_key = api_key or settings.anthropic_api_key
        self._model = model
        self._client: anthropic.Anthropic | None = None

    @property
    def client(self) -> anthropic.Anthropic:
        """Lazy-init Anthropic client."""
        if self._client is None:
            if not self._api_key:
                raise ValueError("ANTHROPIC_API_KEY not set")
            self._client = anthropic.Anthropic(api_key=self._api_key)
        return self._client

    def analyze(
        self,
        analysis_date: date,
        cash: float,
        total_value: float,
        positions: list[dict],
        prices: dict[str, list[float]],
        tickers: list[str],
        indicators: dict[str, dict],
        recent_trades: list[dict],
        regime: str | None = None,
        yield_curve: float | None = None,
        vix: float | None = None,
    ) -> dict:
        """Send market context to Claude and get trade proposals.

        Args:
            analysis_date: Current date.
            cash: Available cash.
            total_value: Total portfolio value.
            positions: Current positions as list of dicts.
            prices: Dict of ticker -> list of recent close prices.
            tickers: Ordered list of tickers to show.
            indicators: Dict of ticker -> indicator values.
            recent_trades: Recent trade history.
            regime: Current regime string.
            yield_curve: 10Y-2Y spread.
            vix: Current VIX.

        Returns:
            Parsed JSON response dict with regime_assessment, reasoning, trades, risk_notes.
        """
        user_prompt = ANALYSIS_PROMPT_TEMPLATE.format(
            date=analysis_date.isoformat(),
            cash=cash,
            total_value=total_value,
            positions_text=build_positions_text(positions),
            price_table=build_price_table(prices, tickers),
            indicators_table=build_indicators_table(indicators),
            macro_context=build_macro_context(regime, yield_curve, vix),
            recent_trades=build_recent_trades_text(recent_trades),
        )

        log.info("agent_calling_claude", model=self._model, date=str(analysis_date))

        response = self.client.messages.create(
            model=self._model,
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )

        # Extract text content
        raw_text = response.content[0].text
        tokens_used = response.usage.input_tokens + response.usage.output_tokens

        log.info(
            "agent_response_received",
            tokens=tokens_used,
            model=response.model,
        )

        # Parse JSON from response
        parsed = self._parse_response(raw_text)
        parsed["_raw"] = raw_text
        parsed["_tokens_used"] = tokens_used
        parsed["_model"] = response.model

        return parsed

    def _parse_response(self, text: str) -> dict:
        """Parse Claude's JSON response, handling potential markdown wrapping.

        Args:
            text: Raw response text.

        Returns:
            Parsed dict. Returns error structure if parsing fails.
        """
        # Strip markdown code fences if present
        cleaned = text.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            log.error("agent_response_parse_failed", error=str(e), raw_text=text[:200])
            return {
                "regime_assessment": "Parse error",
                "reasoning": f"Failed to parse Claude response: {e}",
                "trades": [],
                "risk_notes": "Response was not valid JSON. No trades will be executed.",
            }

    def generate_daily_commentary(
        self,
        analysis_date: date,
        portfolio_a_summary: str,
        portfolio_b_summary: str,
        portfolio_c_summary: str,
        regime: str | None = None,
    ) -> str:
        """Generate a brief daily market commentary for the Telegram brief.

        Args:
            analysis_date: Current date.
            portfolio_a_summary: One-line summary of Portfolio A state.
            portfolio_b_summary: One-line summary of Portfolio B state.
            portfolio_c_summary: One-line summary of Portfolio C state.
            regime: Current detected regime.

        Returns:
            2-3 paragraph commentary string.
        """
        prompt = f"""Date: {analysis_date.isoformat()}
Regime: {regime or 'Unknown'}

Portfolio A (Momentum): {portfolio_a_summary}
Portfolio B (Sector Rotation): {portfolio_b_summary}
Portfolio C (AI Autonomy): {portfolio_c_summary}

Write a concise 2-3 paragraph daily market brief for the Atlas Trading Bot.
Cover: today's key market theme, how each portfolio is positioned, and one thing to watch tomorrow.
Keep it under 200 words. No headers or bullet points — just flowing text."""

        response = self.client.messages.create(
            model=self._model,
            max_tokens=512,
            system="You are a concise financial market commentator for an educational trading bot.",
            messages=[{"role": "user", "content": prompt}],
        )

        return response.content[0].text.strip()
