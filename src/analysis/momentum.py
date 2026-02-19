"""Momentum calculator for Portfolio A.

Calculates 63-day returns (excluding last 5 days) and ranks ETFs.
The skip-last-5-days filter avoids short-term mean reversion drag.
"""

from datetime import date

import pandas as pd
import structlog

from src.storage.models import MomentumRankingRow

log = structlog.get_logger()


def calculate_momentum(
    closes: pd.DataFrame,
    lookback: int = 63,
    skip: int = 5,
) -> pd.DataFrame:
    """Calculate momentum scores for all tickers.

    Momentum = return from T-lookback to T-skip (skipping the most recent days).

    Args:
        closes: DataFrame with tickers as columns and dates as index.
        lookback: Total lookback period in trading days (default 63 ~ 3 months).
        skip: Number of recent days to skip (default 5 ~ 1 week).

    Returns:
        DataFrame with columns: ticker, return_63d, rank.
        Sorted by rank ascending (rank 1 = best momentum).
    """
    # Drop rows where ALL columns are NaN (e.g. weekend rows from BTC-USD 7-day trading)
    closes = closes.dropna(how="all")

    required_rows = lookback + 1
    if len(closes) < required_rows:
        log.warning(
            "insufficient_data_for_momentum",
            rows=len(closes),
            required=required_rows,
        )
        return pd.DataFrame(columns=["ticker", "return_63d", "rank"])

    # Price at T-lookback and T-skip
    price_start = closes.iloc[-(lookback + 1)]
    price_end = closes.iloc[-(skip + 1)] if skip > 0 else closes.iloc[-1]

    returns = (price_end - price_start) / price_start

    # Drop NaN tickers and sort
    returns = returns.dropna().sort_values(ascending=False)

    result = pd.DataFrame(
        {
            "ticker": returns.index,
            "return_63d": returns.values,
            "rank": range(1, len(returns) + 1),
        }
    )

    log.info(
        "momentum_calculated",
        top_ticker=result.iloc[0]["ticker"] if len(result) > 0 else None,
        top_return=round(result.iloc[0]["return_63d"], 4) if len(result) > 0 else None,
        total_tickers=len(result),
    )

    return result


def momentum_to_db_rows(rankings: pd.DataFrame, ranking_date: date) -> list[MomentumRankingRow]:
    """Convert momentum DataFrame to SQLAlchemy model instances.

    Args:
        rankings: DataFrame from calculate_momentum().
        ranking_date: The date these rankings apply to.

    Returns:
        List of MomentumRankingRow ready for database insertion.
    """
    return [
        MomentumRankingRow(
            date=ranking_date,
            ticker=row["ticker"],
            return_63d=row["return_63d"],
            rank=row["rank"],
        )
        for _, row in rankings.iterrows()
    ]


def get_top_n(rankings: pd.DataFrame, n: int = 1) -> list[str]:
    """Get the top N tickers by momentum rank.

    Args:
        rankings: DataFrame from calculate_momentum().
        n: Number of top tickers to return.

    Returns:
        List of ticker symbols.
    """
    return rankings.head(n)["ticker"].tolist()
