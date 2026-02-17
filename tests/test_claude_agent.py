"""Tests for the Claude AI agent — prompt building, response parsing, trade conversion.

All tests use mocked API calls (no real Anthropic requests).
"""

import json
from datetime import date
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from src.agent.claude_agent import (
    ClaudeAgent,
    build_compact_indicators,
    build_compact_price_summary,
    build_indicators_table,
    build_macro_context,
    build_positions_text,
    build_price_table,
    build_recent_trades_text,
    build_system_prompt,
)
from src.strategies.portfolio_b import AIAutonomyStrategy, filter_interesting_tickers

# ── Prompt building tests ────────────────────────────────────────────────────


class TestBuildPositionsText:
    def test_empty_positions(self) -> None:
        result = build_positions_text([])
        assert "no positions" in result

    def test_with_positions(self) -> None:
        positions = [
            {"ticker": "XLK", "shares": 100, "avg_price": 200.0, "market_value": 21000},
            {"ticker": "GLD", "shares": 50, "avg_price": 180.0, "market_value": 9500},
        ]
        result = build_positions_text(positions)
        assert "XLK" in result
        assert "GLD" in result
        assert "100" in result


class TestBuildPriceTable:
    def test_formats_correctly(self) -> None:
        prices = {
            "XLK": [200.0, 201.0, 199.5, 202.0, 203.5],
            "GLD": [180.0, 181.0, 179.0, 182.0, 183.0],
        }
        result = build_price_table(prices, ["XLK", "GLD"])
        assert "XLK" in result
        assert "GLD" in result
        assert "203.50" in result

    def test_skips_short_data(self) -> None:
        prices = {"XLK": [200.0, 201.0]}  # less than 5
        result = build_price_table(prices, ["XLK"])
        assert "XLK" not in result.split("\n")[-1]  # not in data rows


class TestBuildIndicatorsTable:
    def test_formats_indicators(self) -> None:
        indicators = {
            "XLK": {"rsi_14": 55.3, "macd": 1.25, "sma_20": 200.0, "sma_50": 195.0},
        }
        result = build_indicators_table(indicators)
        assert "55.3" in result
        assert "XLK" in result

    def test_handles_none_values(self) -> None:
        indicators = {"XLK": {"rsi_14": None, "macd": None, "sma_20": None, "sma_50": None}}
        result = build_indicators_table(indicators)
        assert "N/A" in result


class TestBuildMacroContext:
    def test_all_data(self) -> None:
        result = build_macro_context(regime="BULL", yield_curve=1.5, vix=18.5)
        assert "BULL" in result
        assert "1.50" in result
        assert "18.5" in result

    def test_inverted_curve(self) -> None:
        result = build_macro_context(yield_curve=-0.3)
        assert "INVERTED" in result

    def test_no_data(self) -> None:
        result = build_macro_context()
        assert "no macro data" in result


class TestBuildRecentTrades:
    def test_empty(self) -> None:
        assert "no recent trades" in build_recent_trades_text([])

    def test_with_trades(self) -> None:
        trades = [{"ticker": "XLK", "side": "BUY", "shares": 100, "price": 200.0, "reason": "test"}]
        result = build_recent_trades_text(trades)
        assert "BUY" in result
        assert "XLK" in result


# ── Agent response parsing tests ─────────────────────────────────────────────


class TestParseResponse:
    def setup_method(self) -> None:
        self.agent = ClaudeAgent(api_key="fake-key")

    def test_valid_json(self) -> None:
        response = json.dumps(
            {
                "regime_assessment": "Bullish",
                "reasoning": "Markets are strong",
                "trades": [{"ticker": "XLK", "side": "BUY", "weight": 0.15, "reason": "momentum"}],
                "risk_notes": "None",
            }
        )
        result = self.agent._parse_response(response)
        assert result["regime_assessment"] == "Bullish"
        assert len(result["trades"]) == 1

    def test_json_with_markdown_fences(self) -> None:
        response = '```json\n{"regime_assessment": "test", "reasoning": "", "trades": [], "risk_notes": ""}\n```'
        result = self.agent._parse_response(response)
        assert result["regime_assessment"] == "test"

    def test_invalid_json_returns_error(self) -> None:
        result = self.agent._parse_response("this is not json at all")
        assert result["regime_assessment"] == "Parse error"
        assert result["trades"] == []


# ── Trade conversion tests ───────────────────────────────────────────────────


class TestAgentResponseToTrades:
    def setup_method(self) -> None:
        self.strategy = AIAutonomyStrategy(agent=ClaudeAgent(api_key="fake-key"))
        self.prices = pd.Series(
            {
                "XLK": 200.0,
                "GLD": 180.0,
                "XLF": 40.0,
                "AAPL": 220.0,
                "MSFT": 410.0,
            }
        )

    def test_basic_buy(self) -> None:
        response = {
            "trades": [{"ticker": "XLK", "side": "BUY", "weight": 0.15, "reason": "momentum"}],
        }
        trades = self.strategy.agent_response_to_trades(
            response,
            total_value=66_000.0,
            current_positions={},
            latest_prices=self.prices,
        )
        assert len(trades) == 1
        assert trades[0].side.value == "BUY"
        assert trades[0].ticker == "XLK"
        # 15% of $66,000 = $9,900, /200 = 49 shares
        assert trades[0].shares == 49.0

    def test_weight_capped_at_30_pct(self) -> None:
        response = {
            "trades": [{"ticker": "XLK", "side": "BUY", "weight": 0.50, "reason": "all in"}],
        }
        trades = self.strategy.agent_response_to_trades(
            response,
            total_value=66_000.0,
            current_positions={},
            latest_prices=self.prices,
        )
        # Should be capped at 30% = $19,800 / 200 = 99 shares
        assert trades[0].shares == 99.0

    def test_invalid_ticker_skipped(self) -> None:
        response = {
            "trades": [{"ticker": "FAKE_TICKER", "side": "BUY", "weight": 0.10, "reason": "fake"}],
        }
        trades = self.strategy.agent_response_to_trades(
            response,
            total_value=66_000.0,
            current_positions={},
            latest_prices=self.prices,
        )
        assert len(trades) == 0

    def test_sell_full_exit(self) -> None:
        response = {
            "trades": [{"ticker": "XLK", "side": "SELL", "weight": 0.0, "reason": "exit"}],
        }
        trades = self.strategy.agent_response_to_trades(
            response,
            total_value=66_000.0,
            current_positions={"XLK": 50.0},
            latest_prices=self.prices,
        )
        assert len(trades) == 1
        assert trades[0].side.value == "SELL"
        assert trades[0].shares == 50.0

    def test_no_trades_proposed(self) -> None:
        response = {"trades": []}
        trades = self.strategy.agent_response_to_trades(
            response,
            total_value=66_000.0,
            current_positions={},
            latest_prices=self.prices,
        )
        assert len(trades) == 0

    def test_empty_trades_on_parse_error(self) -> None:
        # Simulates what happens when Claude returns invalid JSON
        response = {
            "regime_assessment": "Parse error",
            "reasoning": "Failed to parse",
            "trades": [],
            "risk_notes": "",
        }
        trades = self.strategy.agent_response_to_trades(
            response,
            total_value=66_000.0,
            current_positions={},
            latest_prices=self.prices,
        )
        assert len(trades) == 0

    def test_multiple_trades(self) -> None:
        response = {
            "trades": [
                {"ticker": "XLK", "side": "BUY", "weight": 0.15, "reason": "tech"},
                {"ticker": "GLD", "side": "BUY", "weight": 0.10, "reason": "hedge"},
                {"ticker": "XLF", "side": "BUY", "weight": 0.10, "reason": "financials"},
            ],
        }
        trades = self.strategy.agent_response_to_trades(
            response,
            total_value=66_000.0,
            current_positions={},
            latest_prices=self.prices,
        )
        assert len(trades) == 3
        tickers = {t.ticker for t in trades}
        assert tickers == {"XLK", "GLD", "XLF"}

    def test_conviction_high_full_weight(self) -> None:
        """High conviction applies 100% of weight."""
        response = {
            "trades": [
                {"ticker": "XLK", "side": "BUY", "weight": 0.15, "conviction": "high", "reason": "strong trend"},
            ],
        }
        trades = self.strategy.agent_response_to_trades(
            response,
            total_value=66_000.0,
            current_positions={},
            latest_prices=self.prices,
        )
        # 15% of $66K = $9,900 / $200 = 49 shares
        assert trades[0].shares == 49.0

    def test_conviction_medium_70pct(self) -> None:
        """Medium conviction scales weight to 70%."""
        response = {
            "trades": [
                {"ticker": "XLK", "side": "BUY", "weight": 0.20, "conviction": "medium", "reason": "decent setup"},
            ],
        }
        trades = self.strategy.agent_response_to_trades(
            response,
            total_value=66_000.0,
            current_positions={},
            latest_prices=self.prices,
        )
        # 20% * 0.7 = 14% of $66K = $9,240 / $200 = 46 shares
        assert trades[0].shares == 46.0

    def test_conviction_low_40pct(self) -> None:
        """Low conviction scales weight to 40%."""
        response = {
            "trades": [
                {"ticker": "XLK", "side": "BUY", "weight": 0.20, "conviction": "low", "reason": "speculative"},
            ],
        }
        trades = self.strategy.agent_response_to_trades(
            response,
            total_value=66_000.0,
            current_positions={},
            latest_prices=self.prices,
        )
        # 20% * 0.4 = 8% of $66K = $5,280 / $200 = 26 shares
        assert trades[0].shares == 26.0

    def test_conviction_missing_defaults_high(self) -> None:
        """Missing conviction field defaults to high (100%)."""
        response = {
            "trades": [
                {"ticker": "XLK", "side": "BUY", "weight": 0.15, "reason": "no conviction field"},
            ],
        }
        trades = self.strategy.agent_response_to_trades(
            response,
            total_value=66_000.0,
            current_positions={},
            latest_prices=self.prices,
        )
        # Same as high: 15% of $66K = $9,900 / $200 = 49 shares
        assert trades[0].shares == 49.0

    def test_custom_universe_validates_tickers(self) -> None:
        """Custom universe restricts valid tickers for trade validation."""
        response = {
            "trades": [
                {"ticker": "XLK", "side": "BUY", "weight": 0.15, "reason": "ok"},
                {"ticker": "AAPL", "side": "BUY", "weight": 0.10, "reason": "excluded"},
            ],
        }
        # Only XLK and GLD in custom universe — AAPL should be rejected
        trades = self.strategy.agent_response_to_trades(
            response,
            total_value=66_000.0,
            current_positions={},
            latest_prices=self.prices,
            universe=["XLK", "GLD"],
        )
        assert len(trades) == 1
        assert trades[0].ticker == "XLK"


# ── Decision persistence test ────────────────────────────────────────────────


# ── Compact format tests ────────────────────────────────────────────────────


def _make_closes(tickers: list[str], days: int = 100) -> pd.DataFrame:
    """Generate synthetic close prices for testing compact builders."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range(end="2026-02-05", periods=days)
    data = {}
    for t in tickers:
        base = rng.uniform(50, 300)
        returns = rng.normal(0.0005, 0.015, days)
        prices = base * np.cumprod(1 + returns)
        data[t] = prices
    return pd.DataFrame(data, index=dates)


class TestBuildCompactPriceSummary:
    def test_csv_format(self) -> None:
        closes = _make_closes(["XLK", "GLD", "AAPL"], days=30)
        result = build_compact_price_summary(closes, ["XLK", "GLD"])
        lines = result.strip().split("\n")
        assert lines[0] == "Ticker,Price,1d%,5d%,20d%"
        assert len(lines) == 3  # header + 2 tickers
        assert "XLK" in lines[1]

    def test_percentage_changes(self) -> None:
        closes = _make_closes(["XLK"], days=30)
        result = build_compact_price_summary(closes, ["XLK"])
        lines = result.strip().split("\n")
        parts = lines[1].split(",")
        assert len(parts) == 5
        # Check format: ticker, price, 1d%, 5d%, 20d%
        assert parts[0] == "XLK"
        float(parts[1])  # price is valid float
        float(parts[2])  # percentages are valid floats

    def test_skips_missing_ticker(self) -> None:
        closes = _make_closes(["XLK"], days=30)
        result = build_compact_price_summary(closes, ["XLK", "FAKE"])
        lines = result.strip().split("\n")
        assert len(lines) == 2  # header + 1 ticker

    def test_short_data_handles_gracefully(self) -> None:
        closes = _make_closes(["XLK"], days=3)
        result = build_compact_price_summary(closes, ["XLK"])
        # Should still work, just with 0% for missing lookback periods
        assert "XLK" in result

    def test_much_smaller_than_verbose(self) -> None:
        tickers = ["XLK", "XLF", "GLD", "QQQ", "AAPL", "MSFT", "NVDA", "XLE"]
        closes = _make_closes(tickers, days=30)
        compact = build_compact_price_summary(closes, tickers)
        # Build verbose version for comparison
        prices = {}
        for t in tickers:
            vals = closes[t].dropna().tail(5).tolist()
            if len(vals) >= 5:
                prices[t] = vals
        verbose = build_price_table(prices, tickers)
        # Compact should be significantly smaller
        assert len(compact) < len(verbose)


class TestBuildCompactIndicators:
    def test_csv_format(self) -> None:
        closes = _make_closes(["XLK", "GLD"], days=50)
        result = build_compact_indicators(closes, ["XLK", "GLD"])
        lines = result.strip().split("\n")
        assert lines[0] == "Ticker,RSI,MACD"
        assert len(lines) >= 2  # header + at least 1 ticker

    def test_rsi_and_macd_values(self) -> None:
        closes = _make_closes(["XLK"], days=50)
        result = build_compact_indicators(closes, ["XLK"])
        lines = result.strip().split("\n")
        parts = lines[1].split(",")
        assert parts[0] == "XLK"
        rsi = float(parts[1])
        assert 0 <= rsi <= 100

    def test_skips_short_data(self) -> None:
        closes = _make_closes(["XLK"], days=10)
        result = build_compact_indicators(closes, ["XLK"])
        lines = result.strip().split("\n")
        assert len(lines) == 1  # header only

    def test_much_smaller_than_verbose(self) -> None:
        tickers = ["XLK", "XLF", "GLD", "QQQ", "AAPL"]
        closes = _make_closes(tickers, days=60)
        compact = build_compact_indicators(closes, tickers)
        # Build verbose version
        from src.analysis.technical import compute_all_indicators

        indicators = {}
        for t in tickers:
            ind = compute_all_indicators(closes[t].dropna())
            latest = ind.iloc[-1]
            indicators[t] = {
                "rsi_14": float(latest["rsi_14"]) if pd.notna(latest["rsi_14"]) else None,
                "macd": float(latest["macd"]) if pd.notna(latest["macd"]) else None,
                "sma_20": float(latest["sma_20"]) if pd.notna(latest["sma_20"]) else None,
                "sma_50": float(latest["sma_50"]) if pd.notna(latest["sma_50"]) else None,
            }
        verbose = build_indicators_table(indicators)
        assert len(compact) < len(verbose)


class TestFilterInterestingTickers:
    def test_includes_current_holdings(self) -> None:
        closes = _make_closes(["XLK", "XLF", "GLD", "QQQ", "AAPL"], days=50)
        result = filter_interesting_tickers(closes, ["XLK", "GLD"])
        assert "XLK" in result
        assert "GLD" in result

    def test_includes_top_movers(self) -> None:
        closes = _make_closes(["XLK", "XLF", "GLD", "QQQ", "AAPL"], days=50)
        result = filter_interesting_tickers(closes, [], top_movers=3)
        assert len(result) >= 3

    def test_fewer_than_full_universe(self) -> None:
        from config.universe import PORTFOLIO_B_UNIVERSE

        tickers = [t for t in PORTFOLIO_B_UNIVERSE[:20]]
        closes = _make_closes(tickers, days=50)
        result = filter_interesting_tickers(closes, [])
        # Should be a subset (filtered), not the full universe
        assert len(result) <= len(tickers)

    def test_returns_list(self) -> None:
        closes = _make_closes(["XLK", "XLF", "GLD"], days=50)
        result = filter_interesting_tickers(closes, [])
        assert isinstance(result, list)

    def test_handles_short_data(self) -> None:
        closes = _make_closes(["XLK", "XLF"], days=1)
        result = filter_interesting_tickers(closes, ["XLK"])
        # With only 1 day, can't compute % change — just returns all tickers
        assert isinstance(result, list)

    def test_custom_universe(self) -> None:
        """Custom universe restricts which tickers are considered."""
        closes = _make_closes(["XLK", "XLF", "GLD", "QQQ", "AAPL"], days=50)
        # Only consider XLK and GLD — others should be excluded
        result = filter_interesting_tickers(
            closes,
            [],
            universe=["XLK", "GLD"],
        )
        for t in result:
            assert t in ["XLK", "GLD"]

    def test_custom_universe_includes_holdings(self) -> None:
        """Holdings are included even if in a custom universe."""
        closes = _make_closes(["XLK", "XLF", "GLD"], days=50)
        result = filter_interesting_tickers(
            closes,
            ["XLF"],
            universe=["XLK", "XLF", "GLD"],
        )
        assert "XLF" in result


# ── Decision persistence test ────────────────────────────────────────────────


class TestModelOverride:
    def test_model_override_used(self) -> None:
        """When model_override is provided, it should be used instead of default."""
        agent = ClaudeAgent(api_key="fake-key", model="claude-sonnet-4-6")
        mock_client = MagicMock()
        mock_response = MagicMock()
        resp_json = '{"regime_assessment":"test","reasoning":"","trades":[],"risk_notes":""}'
        mock_response.content = [MagicMock(text=resp_json)]
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 50
        mock_response.model = "claude-opus-4-6"
        mock_client.messages.create.return_value = mock_response
        agent._client = mock_client

        agent.analyze(
            analysis_date=date(2026, 2, 5),
            cash=66_000.0,
            total_value=66_000.0,
            positions=[],
            prices={"XLK": [200.0, 201.0, 199.5, 202.0, 203.5]},
            tickers=["XLK"],
            indicators={},
            recent_trades=[],
            model_override="claude-opus-4-6",
        )

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["model"] == "claude-opus-4-6"

    def test_no_override_uses_default(self) -> None:
        """Without model_override, the default model should be used."""
        agent = ClaudeAgent(api_key="fake-key", model="claude-sonnet-4-6")
        mock_client = MagicMock()
        mock_response = MagicMock()
        resp_json = '{"regime_assessment":"test","reasoning":"","trades":[],"risk_notes":""}'
        mock_response.content = [MagicMock(text=resp_json)]
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 50
        mock_response.model = "claude-sonnet-4-6"
        mock_client.messages.create.return_value = mock_response
        agent._client = mock_client

        agent.analyze(
            analysis_date=date(2026, 2, 5),
            cash=66_000.0,
            total_value=66_000.0,
            positions=[],
            prices={"XLK": [200.0, 201.0, 199.5, 202.0, 203.5]},
            tickers=["XLK"],
            indicators={},
            recent_trades=[],
        )

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["model"] == "claude-sonnet-4-6"


class TestBuildSystemPrompt:
    def test_default_prompt(self) -> None:
        """Without perf stats, returns base prompt with decision framework."""
        prompt = build_system_prompt()
        assert "Kukulkan" in prompt
        assert "Decision Framework" in prompt
        assert "Hard Rules" in prompt
        assert "Track Record" not in prompt
        assert "$66,000" in prompt  # default allocation
        assert "~55 tickers" in prompt  # default universe size

    def test_with_performance_stats(self) -> None:
        """With perf stats, appends track record section."""
        perf = "Portfolio B Performance:\n  Value: $69,300.00 (+5.00%)"
        prompt = build_system_prompt(performance_stats=perf)
        assert "Track Record" in prompt
        assert "$69,300.00" in prompt

    def test_custom_allocation(self) -> None:
        """Custom portfolio allocation replaces default $66K in prompt."""
        prompt = build_system_prompt(portfolio_allocation=100_000.0)
        assert "$100,000" in prompt
        assert "$66,000" not in prompt

    def test_custom_universe_size(self) -> None:
        """Custom universe size replaces default ~55 in prompt."""
        prompt = build_system_prompt(universe_size=42)
        assert "~42 tickers" in prompt
        assert "~55 tickers" not in prompt

    def test_custom_allocation_and_universe(self) -> None:
        """Both allocation and universe size can be customized."""
        prompt = build_system_prompt(
            portfolio_allocation=50_000.0,
            universe_size=30,
        )
        assert "$50,000" in prompt
        assert "~30 tickers" in prompt

    def test_system_prompt_passed_to_api(self) -> None:
        """Custom system prompt reaches the API call."""
        agent = ClaudeAgent(api_key="fake-key")
        mock_client = MagicMock()
        mock_response = MagicMock()
        resp_json = '{"regime_assessment":"test","reasoning":"","trades":[],"risk_notes":""}'
        mock_response.content = [MagicMock(text=resp_json)]
        mock_response.usage.input_tokens = 100
        mock_response.usage.output_tokens = 50
        mock_response.model = "test"
        mock_client.messages.create.return_value = mock_response
        agent._client = mock_client

        custom = "You are a custom prompt."
        agent.analyze(
            analysis_date=date(2026, 2, 5),
            cash=66_000.0,
            total_value=66_000.0,
            positions=[],
            prices={"XLK": [200.0, 201.0, 199.5, 202.0, 203.5]},
            tickers=["XLK"],
            indicators={},
            recent_trades=[],
            system_prompt=custom,
        )

        call_kwargs = mock_client.messages.create.call_args[1]
        assert call_kwargs["system"] == custom


class TestSaveDecision:
    async def test_saves_to_db(self) -> None:
        from src.storage.database import Database

        db = Database(url="sqlite+aiosqlite:///:memory:")
        await db.init_db()

        strategy = AIAutonomyStrategy(agent=ClaudeAgent(api_key="fake-key"))
        response = {
            "regime_assessment": "Bullish outlook",
            "reasoning": "Markets look strong",
            "trades": [],
            "risk_notes": "Low risk",
            "_model": "claude-sonnet-4-6",
            "_tokens_used": 1500,
        }
        await strategy.save_decision(db, date(2026, 2, 5), response, [])

        # Verify it was saved
        from sqlalchemy import select

        from src.storage.models import AgentDecisionRow

        async with db.session() as s:
            result = await s.execute(select(AgentDecisionRow))
            rows = result.scalars().all()
        assert len(rows) == 1
        assert rows[0].reasoning == "Markets look strong"
        assert rows[0].tokens_used == 1500

        await db.close()
