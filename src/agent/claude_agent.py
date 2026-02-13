"""Claude AI agent for trade analysis and Portfolio B decision-making.

Uses Anthropic Claude API (Sonnet 4.5) to:
- Analyze market conditions and news sentiment
- Generate trade proposals for Portfolio B
- Provide reasoning for the daily brief
"""

import json
from datetime import date

import anthropic
import pandas as pd
import structlog

from config.settings import settings
from config.strategies import PORTFOLIO_B
from src.agent.strategy_directives import SESSION_DIRECTIVES, STRATEGY_MAP

log = structlog.get_logger()


def _build_base_prompt(
    portfolio_allocation: float = 66_000.0,
    universe_size: int = 55,
) -> str:
    """Build the base system prompt with dynamic allocation and universe size."""
    return (
        f"You are Kukulkan, an AI portfolio manager for an "
        f"educational paper trading bot.\n"
        f"You manage Portfolio B (${portfolio_allocation:,.0f} virtual allocation) "
        f"with full autonomy "
        f"over a universe of ~{universe_size} tickers including sector ETFs, "
        f"thematic ETFs, "
        f"inverse ETFs, commodities, individual stocks, and a Bitcoin proxy.\n\n"
        f"Your goals:\n"
        f"1. Maximize risk-adjusted returns over the medium term (weeks to months)\n"
        f"2. Demonstrate thoughtful portfolio construction with clear reasoning\n"
        f"3. Consider macro regime, sector rotation, momentum, valuation, and sentiment\n"
        f"4. Manage risk through diversification and position sizing\n\n"
        f"Constraints:\n"
        f"- Maximum 10 positions at any time\n"
        f"- No single position > 30% of portfolio value\n"
        f"- You must output valid JSON in the specified format\n\n"
        f"Be contrarian when the data supports it. Avoid herding into crowded trades.\n"
        f"Think in terms of risk/reward asymmetry, not just direction."
    )


_DEFAULT_SYSTEM_PROMPT = _build_base_prompt()

# Keep backward-compatible alias
SYSTEM_PROMPT = _DEFAULT_SYSTEM_PROMPT


def build_system_prompt(
    performance_stats: str | None = None,
    memory_context: str | None = None,
    strategy_mode: str = "conservative",
    session: str = "",
    regime_summary: str | None = None,
    portfolio_allocation: float | None = None,
    universe_size: int | None = None,
    trailing_stops_context: str | None = None,
    earnings_context: str | None = None,
    watchlist_context: str | None = None,
) -> str:
    """Build an enhanced system prompt with performance context and memory.

    Prompt assembly priority: Regime > Session > Strategy > Performance > Memory.

    Args:
        performance_stats: Pre-formatted performance text from PerformanceTracker.
        memory_context: Pre-formatted memory text from AgentMemoryManager.
        strategy_mode: One of "conservative", "standard", "aggressive".
        session: Current session name ("Morning", "Midday", "Closing", or "").
        regime_summary: One-line regime description from RegimeClassifier.
        portfolio_allocation: Dollar allocation for Portfolio B (dynamic per tenant).
        universe_size: Number of tickers in the tenant's universe.

    Returns:
        Full system prompt string.
    """
    if portfolio_allocation is not None or universe_size is not None:
        prompt = _build_base_prompt(
            portfolio_allocation=portfolio_allocation or 66_000.0,
            universe_size=universe_size or 55,
        )
    else:
        prompt = _DEFAULT_SYSTEM_PROMPT

    prompt += """

Decision Framework:
1. ASSESS regime: Is the market risk-on, risk-off, or transitioning?
2. CHECK portfolio health: Are you in drawdown? Concentrate or diversify?
3. IDENTIFY opportunities: What has the best risk/reward right now?
4. SIZE positions: Conviction = size. Low conviction = small or skip.
5. MANAGE risk: Never let a single position become an existential threat.

Hard Rules:
- If drawdown > 10%, reduce gross exposure and tighten stops mentally.
- If VIX > 30, hold at least 20% cash or inverse/hedge exposure.
- IBIT (Bitcoin proxy): treat as a momentum/sentiment signal, not a core holding.
  Size max 10% unless strong trend confirmation.
- Avoid round-tripping: don't sell and rebuy the same ticker within 3 days."""

    # 1. Regime summary (highest priority)
    if regime_summary:
        prompt += f"\n\n## Current Market Regime\n{regime_summary}"

    # 2. Session directive (skipped when session="" i.e. backtest)
    session_text = SESSION_DIRECTIVES.get(session)
    if session_text:
        prompt += session_text

    # 3. Strategy directive (base philosophy)
    directive = STRATEGY_MAP.get(strategy_mode)
    if directive:
        prompt += f"\n{directive}"

    # 4. Performance stats
    if performance_stats:
        prompt += f"\n\n## Your Track Record\n{performance_stats}"

    # 5. Memory context
    if memory_context:
        prompt += f"\n\n{memory_context}"

    # 6. Trailing stops context
    if trailing_stops_context:
        prompt += f"\n\n## Active Trailing Stops\n{trailing_stops_context}"

    # 7. Earnings context
    if earnings_context:
        prompt += f"\n\n## Upcoming Earnings\n{earnings_context}"

    # 8. Watchlist context
    if watchlist_context:
        prompt += f"\n\n## Your Watchlist\n{watchlist_context}"

    return prompt


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

## News Intelligence (filtered & ranked by relevance)
Format: TICKER|SIGNAL|EVENT|#SOURCES
Signal types: POS=bullish, NEG=bearish, MACRO=regime-relevant, EVENT=corporate action, INFO=neutral
{news_context}

Interpret: More sources (#SRC) = higher confidence. POS/NEG on held tickers = review position.
MACRO signals should inform your regime assessment.

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
      "conviction": "high",
      "reason": "brief reason for this specific trade"
    }}
  ],
  "risk_notes": "Any risk concerns or hedging rationale",
  "suggested_tickers": [
    {{
      "ticker": "PLTR",
      "rationale": "brief reason this ticker should be added to the universe"
    }}
  ],
  "memory_notes": [
    {{
      "key": "thesis-tech",
      "content": "XLK showing relative strength vs SPY, tech rotation thesis intact"
    }}
  ],
  "watchlist_updates": [
    {{
      "action": "add",
      "ticker": "PLTR",
      "reason": "AI sector rotation accelerating",
      "conviction": "medium",
      "target_entry": 22.50
    }}
  ]
}}

Rules for the trades array:
- "side" must be "BUY" or "SELL"
- "weight" is target portfolio weight (0.0 to 0.30). Use 0.0 to fully exit a position.
- "conviction": "high", "medium", or "low" (default: "high" if omitted)
  Conviction adjusts effective weight: high=100%, medium=70%, low=40%
- Only include tickers you want to change. Omit tickers you want to hold unchanged.
- If no trades needed, return an empty trades array.
- Ticker must be from the universe provided in the price table.

Rules for suggested_tickers:
- Only suggest tickers NOT already in your universe.
- Include only if you have strong conviction based on the news or market conditions.
- If no suggestions, return an empty array or omit the field.
- Suggestions will be validated and require human approval before being tradeable.

Rules for memory_notes:
- Use memory_notes to persist observations you want to remember across sessions.
- Each note has a "key" (e.g. "thesis-tech") and "content" (max ~50 words).
- You can overwrite a previous note by reusing the same key.
- Max 10 notes stored. Use for: theses, lessons, correlations, timing insights.
- If nothing worth remembering, return an empty array or omit the field.

Rules for watchlist_updates:
- "action" must be "add" or "remove"
- "conviction": "high", "medium", or "low"
- "target_entry": optional target entry price
- Added items expire after 14 days if not acted on
- When you include a watchlist ticker in your trades, it auto-removes from watchlist
- Max 10 watchlist items. If full, remove lowest conviction before adding.
- If no changes needed, return an empty array or omit the field."""


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
            lines.append(f"{t:<8} {vals[0]:>8.2f} {vals[1]:>8.2f} {vals[2]:>8.2f} {vals[3]:>8.2f} {vals[4]:>8.2f}")
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
        lines.append(f"- Market Regime: **{regime}**")
    if yield_curve is not None:
        curve_status = "INVERTED" if yield_curve < 0 else "normal"
        lines.append(f"- Yield Curve (10Y-2Y): {yield_curve:+.2f}% ({curve_status})")
    if vix is not None:
        vix_status = "HIGH FEAR" if vix > 30 else "elevated" if vix > 20 else "low"
        lines.append(f"- VIX: {vix:.1f} ({vix_status})")
    return "\n".join(lines) if lines else "  (no macro data available)"


def build_compact_price_summary(
    closes: pd.DataFrame,
    tickers: list[str],
) -> str:
    """Build a compact CSV price summary with percentage changes.

    Shows current price plus 1d%, 5d%, 20d% changes instead of raw OHLCV.
    ~75% smaller than the verbose 5-day price table.

    Args:
        closes: DataFrame of close prices (tickers as columns, dates as index).
        tickers: List of tickers to include.

    Returns:
        CSV-formatted string.
    """
    lines = ["Ticker,Price,1d%,5d%,20d%"]
    for t in tickers:
        if t not in closes.columns:
            continue
        series = closes[t].dropna()
        if len(series) < 2:
            continue
        price = series.iloc[-1]
        pct_1d = ((series.iloc[-1] / series.iloc[-2]) - 1) * 100 if len(series) >= 2 else 0
        pct_5d = ((series.iloc[-1] / series.iloc[-6]) - 1) * 100 if len(series) >= 6 else 0
        pct_20d = ((series.iloc[-1] / series.iloc[-21]) - 1) * 100 if len(series) >= 21 else 0
        lines.append(f"{t},{price:.2f},{pct_1d:+.1f},{pct_5d:+.1f},{pct_20d:+.1f}")
    return "\n".join(lines)


def build_compact_indicators(
    closes: pd.DataFrame,
    tickers: list[str],
) -> str:
    """Build a compact CSV of RSI + MACD for interesting tickers only.

    ~75% smaller than the full 4-column indicator table.

    Args:
        closes: DataFrame of close prices.
        tickers: Interesting tickers to include.

    Returns:
        CSV-formatted string.
    """
    from src.analysis.technical import compute_macd, compute_rsi

    lines = ["Ticker,RSI,MACD"]
    for t in tickers:
        if t not in closes.columns:
            continue
        series = closes[t].dropna()
        if len(series) < 30:
            continue
        try:
            rsi = compute_rsi(series)
            macd_df = compute_macd(series)
            rsi_val = rsi.iloc[-1]
            macd_val = macd_df["macd"].iloc[-1]
            if pd.notna(rsi_val) and pd.notna(macd_val):
                lines.append(f"{t},{rsi_val:.1f},{macd_val:.2f}")
        except Exception:
            continue
    return "\n".join(lines)


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
        reason = t.get("reason", "")
        lines.append(f"  {t['side']} {t['shares']:.0f}x {t['ticker']} @ ${t['price']:.2f} — {reason}")
    return "\n".join(lines)


class ClaudeAgent:
    """Claude-powered trading agent for Portfolio B."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = PORTFOLIO_B.model,
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
        news_context: str = "",
        interesting_tickers: list[str] | None = None,
        closes_df: pd.DataFrame | None = None,
        model_override: str | None = None,
        system_prompt: str | None = None,
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
            news_context: Formatted news headlines for context.
            interesting_tickers: If provided, use compact format with these tickers.
            closes_df: Full closes DataFrame for compact builders.
            model_override: If provided, use this model instead of the default.
            system_prompt: If provided, use this system prompt instead of the default.

        Returns:
            Parsed JSON response dict with regime_assessment, reasoning, trades, risk_notes.
        """
        # Use compact format when interesting_tickers and closes_df provided
        if interesting_tickers is not None and closes_df is not None:
            price_section = build_compact_price_summary(closes_df, interesting_tickers)
            indicator_section = build_compact_indicators(closes_df, interesting_tickers)
        else:
            price_section = build_price_table(prices, tickers)
            indicator_section = build_indicators_table(indicators)

        user_prompt = ANALYSIS_PROMPT_TEMPLATE.format(
            date=analysis_date.isoformat(),
            cash=cash,
            total_value=total_value,
            positions_text=build_positions_text(positions),
            price_table=price_section,
            indicators_table=indicator_section,
            macro_context=build_macro_context(regime, yield_curve, vix),
            recent_trades=build_recent_trades_text(recent_trades),
            news_context=news_context or "  (no recent news available)",
        )

        effective_model = model_override or self._model
        effective_prompt = system_prompt or SYSTEM_PROMPT
        log.info("agent_calling_claude", model=effective_model, date=str(analysis_date))

        response = self.client.messages.create(
            model=effective_model,
            max_tokens=2048,
            system=effective_prompt,
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
        regime: str | None = None,
    ) -> str:
        """Generate a brief daily market commentary for the Telegram brief.

        Args:
            analysis_date: Current date.
            portfolio_a_summary: One-line summary of Portfolio A state.
            portfolio_b_summary: One-line summary of Portfolio B state.
            regime: Current detected regime.

        Returns:
            2-3 paragraph commentary string.
        """
        prompt = f"""Date: {analysis_date.isoformat()}
Regime: {regime or "Unknown"}

Portfolio A (Momentum): {portfolio_a_summary}
Portfolio B (AI Autonomy): {portfolio_b_summary}

Write a concise 2-3 paragraph daily market brief for the Kukulkan Trading Bot.
Cover: today's key market theme, how each portfolio is positioned, and one thing to watch tomorrow.
Keep it under 200 words. No headers or bullet points — just flowing text."""

        response = self.client.messages.create(
            model=self._model,
            max_tokens=512,
            system="You are a concise financial market commentator for an educational trading bot.",
            messages=[{"role": "user", "content": prompt}],
        )

        return response.content[0].text.strip()
