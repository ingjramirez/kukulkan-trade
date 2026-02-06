"""Tests for the Claude AI agent — prompt building, response parsing, trade conversion.

All tests use mocked API calls (no real Anthropic requests).
"""

import json
from datetime import date
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.agent.claude_agent import (
    ClaudeAgent,
    build_indicators_table,
    build_macro_context,
    build_positions_text,
    build_price_table,
    build_recent_trades_text,
)
from src.strategies.portfolio_c import AIAutonomyStrategy


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
        response = json.dumps({
            "regime_assessment": "Bullish",
            "reasoning": "Markets are strong",
            "trades": [{"ticker": "XLK", "side": "BUY", "weight": 0.15, "reason": "momentum"}],
            "risk_notes": "None",
        })
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
        self.prices = pd.Series({
            "XLK": 200.0,
            "GLD": 180.0,
            "XLF": 40.0,
            "AAPL": 220.0,
            "MSFT": 410.0,
        })

    def test_basic_buy(self) -> None:
        response = {
            "trades": [{"ticker": "XLK", "side": "BUY", "weight": 0.15, "reason": "momentum"}],
        }
        trades = self.strategy.agent_response_to_trades(
            response, total_value=33_333.0, current_positions={}, latest_prices=self.prices,
        )
        assert len(trades) == 1
        assert trades[0].side.value == "BUY"
        assert trades[0].ticker == "XLK"
        # 15% of $33,333 = $4,999, /200 = 24 shares
        assert trades[0].shares == 24.0

    def test_weight_capped_at_30_pct(self) -> None:
        response = {
            "trades": [{"ticker": "XLK", "side": "BUY", "weight": 0.50, "reason": "all in"}],
        }
        trades = self.strategy.agent_response_to_trades(
            response, total_value=33_333.0, current_positions={}, latest_prices=self.prices,
        )
        # Should be capped at 30% = $9,999 / 200 = 49 shares
        assert trades[0].shares == 49.0

    def test_invalid_ticker_skipped(self) -> None:
        response = {
            "trades": [{"ticker": "FAKE_TICKER", "side": "BUY", "weight": 0.10, "reason": "fake"}],
        }
        trades = self.strategy.agent_response_to_trades(
            response, total_value=33_333.0, current_positions={}, latest_prices=self.prices,
        )
        assert len(trades) == 0

    def test_sell_full_exit(self) -> None:
        response = {
            "trades": [{"ticker": "XLK", "side": "SELL", "weight": 0.0, "reason": "exit"}],
        }
        trades = self.strategy.agent_response_to_trades(
            response,
            total_value=33_333.0,
            current_positions={"XLK": 50.0},
            latest_prices=self.prices,
        )
        assert len(trades) == 1
        assert trades[0].side.value == "SELL"
        assert trades[0].shares == 50.0

    def test_no_trades_proposed(self) -> None:
        response = {"trades": []}
        trades = self.strategy.agent_response_to_trades(
            response, total_value=33_333.0, current_positions={}, latest_prices=self.prices,
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
            response, total_value=33_333.0, current_positions={}, latest_prices=self.prices,
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
            response, total_value=33_333.0, current_positions={}, latest_prices=self.prices,
        )
        assert len(trades) == 3
        tickers = {t.ticker for t in trades}
        assert tickers == {"XLK", "GLD", "XLF"}


# ── Decision persistence test ────────────────────────────────────────────────


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
            "_model": "claude-sonnet-4-5-20250929",
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
