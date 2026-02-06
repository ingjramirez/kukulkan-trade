"""Portfolio A: Aggressive Momentum strategy.

Picks the top 1 ETF by 63-day momentum (skipping last 5 days).
Daily rebalance, no stop-loss.
"""

from datetime import date

import pandas as pd
import structlog

from config.strategies import PORTFOLIO_A
from config.universe import PORTFOLIO_A_UNIVERSE
from src.analysis.momentum import calculate_momentum, get_top_n, momentum_to_db_rows
from src.storage.models import OrderSide, PortfolioName, TradeSchema

log = structlog.get_logger()


class MomentumStrategy:
    """Aggressive momentum: hold the single highest-momentum ETF."""

    def __init__(
        self,
        lookback: int = PORTFOLIO_A.momentum_lookback_days,
        skip: int = PORTFOLIO_A.momentum_skip_days,
        top_n: int = PORTFOLIO_A.top_n,
    ) -> None:
        self._lookback = lookback
        self._skip = skip
        self._top_n = top_n

    def rank(self, closes: pd.DataFrame) -> pd.DataFrame:
        """Calculate momentum rankings for the Portfolio A universe.

        Args:
            closes: DataFrame of close prices (tickers as columns, dates as index).
                    Should contain at least PORTFOLIO_A_UNIVERSE tickers.

        Returns:
            DataFrame with columns: ticker, return_63d, rank.
        """
        # Filter to Portfolio A universe only
        available = [t for t in PORTFOLIO_A_UNIVERSE if t in closes.columns]
        filtered = closes[available]

        rankings = calculate_momentum(
            filtered,
            lookback=self._lookback,
            skip=self._skip,
        )
        return rankings

    def get_target_ticker(self, closes: pd.DataFrame) -> str | None:
        """Determine which single ETF to hold.

        Args:
            closes: DataFrame of close prices.

        Returns:
            Ticker symbol of the top momentum ETF, or None if insufficient data.
        """
        rankings = self.rank(closes)
        if rankings.empty:
            return None
        top = get_top_n(rankings, n=self._top_n)
        return top[0] if top else None

    def generate_trades(
        self,
        closes: pd.DataFrame,
        current_positions: dict[str, float],
        cash: float,
    ) -> list[TradeSchema]:
        """Generate trade signals for a rebalance.

        Args:
            closes: DataFrame of close prices.
            current_positions: Dict of ticker -> shares currently held.
            cash: Available cash in Portfolio A.

        Returns:
            List of TradeSchema objects (sells first, then buys).
        """
        target = self.get_target_ticker(closes)
        if target is None:
            log.warning("no_momentum_target")
            return []

        latest_prices = closes.iloc[-1]
        target_price = latest_prices.get(target)
        if target_price is None or pd.isna(target_price):
            log.warning("no_price_for_target", ticker=target)
            return []

        trades: list[TradeSchema] = []

        # Sell everything we don't want
        for ticker, shares in current_positions.items():
            if ticker != target and shares > 0:
                price = latest_prices.get(ticker)
                if price is not None and not pd.isna(price):
                    trades.append(TradeSchema(
                        portfolio=PortfolioName.A,
                        ticker=ticker,
                        side=OrderSide.SELL,
                        shares=shares,
                        price=float(price),
                        reason=f"rotation out, new target={target}",
                    ))

        # Calculate available cash after sells
        sell_proceeds = sum(t.total for t in trades)
        available = cash + sell_proceeds

        # Buy target if not already holding
        if target not in current_positions or current_positions[target] == 0:
            shares_to_buy = int(available / target_price)
            if shares_to_buy > 0:
                trades.append(TradeSchema(
                    portfolio=PortfolioName.A,
                    ticker=target,
                    side=OrderSide.BUY,
                    shares=float(shares_to_buy),
                    price=float(target_price),
                    reason="momentum rank #1",
                ))

        log.info(
            "portfolio_a_trades",
            target=target,
            num_trades=len(trades),
            available_cash=round(available, 2),
        )
        return trades

    def get_ranking_rows(self, closes: pd.DataFrame, ranking_date: date):
        """Get momentum rankings as DB rows for persistence.

        Args:
            closes: DataFrame of close prices.
            ranking_date: Date for these rankings.

        Returns:
            List of MomentumRankingRow objects.
        """
        rankings = self.rank(closes)
        return momentum_to_db_rows(rankings, ranking_date)
