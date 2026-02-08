"""Backtest runner for historical strategy simulation.

Replays 6 months of trading using pre-fetched market data.
Runs Portfolio A (momentum) and a deterministic MockPortfolioB
through the existing PaperTrader.
"""

from datetime import date

import pandas as pd
import structlog
import yfinance as yf

from config.universe import (
    FULL_UNIVERSE,
    PORTFOLIO_B_UNIVERSE,
)
from src.analysis.technical import compute_rsi
from src.execution.paper_trader import PaperTrader
from src.storage.database import Database
from src.storage.models import (
    Base,
    OrderSide,
    PortfolioName,
    TradeSchema,
)
from src.strategies.portfolio_a import MomentumStrategy

log = structlog.get_logger()


class MockPortfolioB:
    """Deterministic Portfolio B substitute for backtesting.

    Picks top 5 tickers by 20-day momentum with RSI < 70,
    equal-weight allocation. Generates buy/sell trades to rebalance.
    """

    def __init__(self, top_n: int = 5) -> None:
        self._top_n = top_n

    def generate_trades(
        self,
        closes: pd.DataFrame,
        current_positions: dict[str, float],
        cash: float,
        total_value: float,
    ) -> list[TradeSchema]:
        tickers = [t for t in PORTFOLIO_B_UNIVERSE if t in closes.columns]
        if len(closes) < 21:
            return []

        # 20-day momentum
        price_now = closes[tickers].iloc[-1]
        price_20d = closes[tickers].iloc[-21]
        momentum = ((price_now - price_20d) / price_20d).dropna()

        # RSI filter: exclude overbought (RSI > 70)
        valid_tickers = []
        for t in momentum.index:
            if t in closes.columns and len(closes[t].dropna()) >= 20:
                rsi = compute_rsi(closes[t].dropna())
                if not rsi.empty and not pd.isna(rsi.iloc[-1]) and rsi.iloc[-1] < 70:
                    valid_tickers.append(t)

        # Rank by momentum, pick top N
        filtered = momentum[momentum.index.isin(valid_tickers)].sort_values(ascending=False)
        selected = filtered.head(self._top_n).index.tolist()

        if not selected:
            return []

        trades: list[TradeSchema] = []
        latest_prices = closes.iloc[-1]

        # Sell positions not in target
        for ticker, shares in current_positions.items():
            if ticker not in selected and shares > 0:
                price = latest_prices.get(ticker)
                if price is not None and not pd.isna(price):
                    trades.append(TradeSchema(
                        portfolio=PortfolioName.B,
                        ticker=ticker,
                        side=OrderSide.SELL,
                        shares=shares,
                        price=float(price),
                        reason="mock B: rotation out",
                    ))

        # Calculate available after sells
        sell_proceeds = sum(t.total for t in trades)
        available = cash + sell_proceeds

        # Add value of positions we're keeping
        for ticker in selected:
            if ticker in current_positions:
                price = latest_prices.get(ticker, 0)
                if not pd.isna(price):
                    available += current_positions[ticker] * price

        # Equal-weight buys
        target_per = available / len(selected)
        for ticker in selected:
            price = latest_prices.get(ticker)
            if price is None or pd.isna(price) or price <= 0:
                continue
            current_shares = current_positions.get(ticker, 0)
            target_shares = int(target_per / price)
            delta = target_shares - current_shares
            if delta > 0:
                trades.append(TradeSchema(
                    portfolio=PortfolioName.B,
                    ticker=ticker,
                    side=OrderSide.BUY,
                    shares=float(delta),
                    price=float(price),
                    reason="mock B: momentum+RSI rebalance",
                ))
            elif delta < 0:
                trades.append(TradeSchema(
                    portfolio=PortfolioName.B,
                    ticker=ticker,
                    side=OrderSide.SELL,
                    shares=float(abs(delta)),
                    price=float(price),
                    reason="mock B: rebalance trim",
                ))

        return trades


class BacktestRunner:
    """Replays historical data through both portfolio strategies."""

    def __init__(self, db_path: str = "data/backtest.db") -> None:
        self._db_url = f"sqlite+aiosqlite:///{db_path}"
        self._db: Database | None = None

    async def run(
        self,
        months: int = 6,
        clean: bool = False,
        use_ai: bool = False,
        dry_run: bool = False,
        ai_budget: float = 1.50,
        prompt_override: str | None = None,
        strategy_mode: str | None = None,
        run_label: str = "default",
    ) -> dict:
        """Execute the full backtest.

        Args:
            months: Number of months of history to simulate.
            clean: If True, drop and recreate all tables before running.
            use_ai: If True, use real Claude AI for Portfolio B.
            dry_run: If True, estimate token cost without running.
            ai_budget: Maximum USD to spend on AI API calls.
            prompt_override: Custom system prompt text for AI.
            strategy_mode: Strategy persona (conservative/standard/aggressive).
            run_label: Label for this run (e.g. "standard").

        Returns:
            Summary dict with performance metrics.
        """
        self._db = Database(url=self._db_url)

        if clean:
            log.info("backtest_clean_db")
            async with self._db._engine.begin() as conn:
                await conn.run_sync(Base.metadata.drop_all)

        await self._db.init_db()

        log.info("backtest_start", months=months)

        # Step 1: Fetch historical data
        closes, volumes = self._fetch_data()
        if closes.empty:
            log.error("backtest_no_data")
            return {"error": "No market data fetched"}

        log.info("backtest_data_fetched", tickers=len(closes.columns), days=len(closes))

        # Step 2: Determine simulation window
        trading_days = closes.index.tolist()
        sim_days = int(months * 21)  # ~21 trading days per month
        # Need lookback buffer (68 days for Portfolio A momentum)
        lookback_buffer = 200
        start_idx = max(0, len(trading_days) - sim_days - lookback_buffer)
        sim_start_idx = start_idx + lookback_buffer

        if sim_start_idx >= len(trading_days):
            sim_start_idx = min(start_idx + 68, len(trading_days) - 1)

        sim_dates = trading_days[sim_start_idx:]
        log.info(
            "backtest_window",
            sim_days=len(sim_dates),
            start=str(sim_dates[0]),
            end=str(sim_dates[-1]),
        )

        # Dry run: estimate token cost without executing
        if dry_run:
            return self._estimate_cost(len(sim_dates), use_ai)

        # Step 3: Initialize
        trader = PaperTrader(self._db)
        await trader.initialize_portfolios()
        strategy_a = MomentumStrategy()

        # Portfolio B: real AI or mock
        ai_bt_strategy = None
        mock_b = None
        if use_ai:
            from src.backtest.ai_strategy import AIBacktestStrategy
            ai_bt_strategy = AIBacktestStrategy(
                budget_usd=ai_budget,
                run_label=run_label,
                prompt_override=prompt_override,
                strategy_mode=strategy_mode,
            )
            log.info(
                "backtest_using_real_ai",
                budget=f"${ai_budget:.2f}",
                label=run_label,
            )
        else:
            mock_b = MockPortfolioB()

        # Step 4: Day-by-day simulation
        trade_counts = {"A": 0, "B": 0}

        for i, sim_date in enumerate(sim_dates):
            day_idx = trading_days.index(sim_date)
            # Slice data up to current day (no look-forward bias)
            closes_slice = closes.iloc[:day_idx + 1]

            all_trades: list[TradeSchema] = []

            # Portfolio A
            try:
                trades_a = await self._run_portfolio_a(
                    strategy_a, closes_slice, trader, sim_date
                )
                all_trades.extend(trades_a)
                trade_counts["A"] += len(trades_a)
            except Exception as e:
                log.warning("backtest_portfolio_a_error", day=str(sim_date), error=str(e))

            # Portfolio B
            try:
                if use_ai and ai_bt_strategy is not None:
                    trades_b = await self._run_portfolio_b_ai(
                        ai_bt_strategy, closes_slice, volumes,
                        trader, sim_date,
                    )
                elif mock_b is not None:
                    trades_b = await self._run_portfolio_b_mock(
                        mock_b, closes_slice, trader, sim_date
                    )
                else:
                    trades_b = []
                all_trades.extend(trades_b)
                trade_counts["B"] += len(trades_b)
            except Exception as e:
                log.warning("backtest_portfolio_b_error", day=str(sim_date), error=str(e))

            # Execute all trades
            if all_trades:
                await trader.execute_trades(all_trades)

            # Take snapshots
            latest_prices = {
                t: float(closes_slice[t].iloc[-1])
                for t in closes_slice.columns
                if not pd.isna(closes_slice[t].iloc[-1])
            }
            for pname in ("A", "B"):
                await trader.take_snapshot(pname, sim_date, latest_prices)

            if (i + 1) % 20 == 0:
                log.info("backtest_progress", day=i + 1, total=len(sim_dates), date=str(sim_date))

        # Step 5: Compute summary
        summary = await self._compute_summary(trade_counts)

        # Save AI decisions and cost report
        if ai_bt_strategy is not None:
            decisions_path = ai_bt_strategy.save_decisions()
            if decisions_path:
                summary["ai_decisions_path"] = decisions_path
            summary["ai_cost_report"] = ai_bt_strategy.get_cost_report()

        await self._db.close()

        log.info("backtest_complete", **summary)
        return summary

    def _fetch_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Batch-fetch 2 years of OHLCV data for the full universe."""
        tickers = FULL_UNIVERSE
        log.info("backtest_fetching_data", tickers=len(tickers))

        try:
            raw = yf.download(tickers, period="2y", group_by="ticker", progress=False)
        except Exception as e:
            log.error("backtest_yfinance_failed", error=str(e))
            return pd.DataFrame(), pd.DataFrame()

        if raw.empty:
            return pd.DataFrame(), pd.DataFrame()

        closes = pd.DataFrame()
        volumes = pd.DataFrame()

        for t in tickers:
            try:
                if len(tickers) == 1:
                    closes[t] = raw["Close"]
                    volumes[t] = raw["Volume"]
                else:
                    if t in raw.columns.get_level_values(0):
                        closes[t] = raw[t]["Close"]
                        volumes[t] = raw[t]["Volume"]
            except (KeyError, TypeError):
                continue

        # Drop rows where all values are NaN
        closes = closes.dropna(how="all")
        volumes = volumes.dropna(how="all")

        return closes, volumes

    async def _run_portfolio_a(
        self,
        strategy: MomentumStrategy,
        closes: pd.DataFrame,
        trader: PaperTrader,
        sim_date: date,
    ) -> list[TradeSchema]:
        portfolio = await self._db.get_portfolio("A")
        positions = await self._db.get_positions("A")
        position_map = {p.ticker: p.shares for p in positions}
        cash = portfolio.cash if portfolio else 33_000.0

        trades = strategy.generate_trades(closes, position_map, cash)

        # Save momentum rankings
        ranking_rows = strategy.get_ranking_rows(closes, sim_date)
        if ranking_rows:
            await self._db.save_momentum_rankings(ranking_rows)

        return trades

    async def _run_portfolio_b_mock(
        self,
        mock: MockPortfolioB,
        closes: pd.DataFrame,
        trader: PaperTrader,
        sim_date: date,
    ) -> list[TradeSchema]:
        portfolio = await self._db.get_portfolio("B")
        positions = await self._db.get_positions("B")
        position_map = {p.ticker: p.shares for p in positions}
        cash = portfolio.cash if portfolio else 66_000.0
        total_value = portfolio.total_value if portfolio else 66_000.0

        return mock.generate_trades(closes, position_map, cash, total_value)

    async def _run_portfolio_b_ai(
        self,
        ai_bt_strategy,
        closes: pd.DataFrame,
        volumes: pd.DataFrame,
        trader: PaperTrader,
        sim_date: date,
    ) -> list[TradeSchema]:
        """Run Portfolio B using the AI backtest strategy.

        Args:
            ai_bt_strategy: AIBacktestStrategy instance with budget tracking.
            closes: Historical closes up to sim_date.
            volumes: Historical volumes up to sim_date.
            trader: PaperTrader instance.
            sim_date: Current simulation date.

        Returns:
            List of generated trades.
        """
        portfolio = await self._db.get_portfolio("B")
        positions = await self._db.get_positions("B")
        position_map = {p.ticker: p.shares for p in positions}
        cash = portfolio.cash if portfolio else 66_000.0
        total_value = portfolio.total_value if portfolio else 66_000.0

        positions_for_agent = [
            {
                "ticker": p.ticker,
                "shares": p.shares,
                "avg_price": p.avg_price,
                "market_value": (
                    p.shares * float(closes[p.ticker].iloc[-1])
                    if p.ticker in closes.columns else 0
                ),
            }
            for p in positions
        ]

        recent_trades_raw = await self._db.get_trades("B")
        recent_trades = [
            {
                "ticker": t.ticker,
                "side": t.side,
                "shares": t.shares,
                "price": t.price,
                "reason": t.reason or "",
            }
            for t in recent_trades_raw[:5]
        ]

        return ai_bt_strategy.generate_trades(
            closes=closes,
            volumes=volumes,
            positions=positions_for_agent,
            cash=cash,
            total_value=total_value,
            current_positions=position_map,
            recent_trades=recent_trades,
            sim_date=sim_date,
        )

    def _estimate_cost(self, sim_days: int, use_ai: bool) -> dict:
        """Estimate API cost for a backtest run.

        Args:
            sim_days: Number of simulation days.
            use_ai: Whether real AI is being used.

        Returns:
            Summary dict with cost estimate.
        """
        if not use_ai:
            return {
                "dry_run": True,
                "sim_days": sim_days,
                "estimated_api_calls": 0,
                "estimated_tokens": 0,
                "estimated_cost_usd": 0.0,
                "note": "Mock strategy — no API calls needed.",
            }

        # Estimate: ~1500 input + ~500 output tokens per call
        tokens_per_call = 2000
        total_tokens = sim_days * tokens_per_call
        # Sonnet pricing: ~$3/M input + ~$15/M output ≈ $6/M blended
        cost = total_tokens * 6.0 / 1_000_000

        return {
            "dry_run": True,
            "sim_days": sim_days,
            "estimated_api_calls": sim_days,
            "estimated_tokens": total_tokens,
            "estimated_cost_usd": round(cost, 2),
            "note": (
                f"~{sim_days} Claude API calls. "
                f"Estimated ${cost:.2f} at Sonnet rates."
            ),
        }

    async def _compute_summary(self, trade_counts: dict) -> dict:
        """Compute final backtest metrics."""
        summary = {"trade_counts": trade_counts}

        for pname in ("A", "B"):
            snapshots = await self._db.get_snapshots(pname)
            if snapshots:
                values = [s.total_value for s in snapshots]
                start_val = values[0]
                end_val = values[-1]
                total_return = ((end_val - start_val) / start_val) * 100

                # Max drawdown
                peak = values[0]
                max_dd = 0.0
                for v in values:
                    if v > peak:
                        peak = v
                    dd = ((peak - v) / peak) * 100
                    if dd > max_dd:
                        max_dd = dd

                initial = 33_000.0 if pname == "A" else 66_000.0
                summary[f"portfolio_{pname}"] = {
                    "total_return_pct": round(total_return, 2),
                    "max_drawdown_pct": round(max_dd, 2),
                    "final_value": round(end_val, 2),
                    "snapshots": len(snapshots),
                }
            else:
                initial = 33_000.0 if pname == "A" else 66_000.0
                summary[f"portfolio_{pname}"] = {
                    "total_return_pct": 0,
                    "max_drawdown_pct": 0,
                    "final_value": initial,
                    "snapshots": 0,
                }

        return summary
