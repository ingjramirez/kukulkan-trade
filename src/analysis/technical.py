"""Technical analysis wrapper using the `ta` library.

Provides RSI, MACD, SMA, and Bollinger Bands for any ticker.
"""

import pandas as pd
import structlog
from ta.momentum import RSIIndicator
from ta.trend import MACD, SMAIndicator
from ta.volatility import BollingerBands

log = structlog.get_logger()


def compute_rsi(closes: pd.Series, window: int = 14) -> pd.Series:
    """Calculate Relative Strength Index.

    Args:
        closes: Series of close prices.
        window: RSI lookback period (default 14).

    Returns:
        Series of RSI values (0-100).
    """
    return RSIIndicator(close=closes, window=window).rsi()


def compute_macd(
    closes: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """Calculate MACD, signal line, and histogram.

    Args:
        closes: Series of close prices.
        fast: Fast EMA period.
        slow: Slow EMA period.
        signal: Signal line EMA period.

    Returns:
        DataFrame with columns: macd, signal, histogram.
    """
    indicator = MACD(close=closes, window_fast=fast, window_slow=slow, window_sign=signal)
    return pd.DataFrame(
        {
            "macd": indicator.macd(),
            "signal": indicator.macd_signal(),
            "histogram": indicator.macd_diff(),
        }
    )


def compute_sma(closes: pd.Series, window: int = 20) -> pd.Series:
    """Calculate Simple Moving Average.

    Args:
        closes: Series of close prices.
        window: SMA lookback period.

    Returns:
        Series of SMA values.
    """
    return SMAIndicator(close=closes, window=window).sma_indicator()


def compute_bollinger_bands(closes: pd.Series, window: int = 20, std_dev: float = 2.0) -> pd.DataFrame:
    """Calculate Bollinger Bands.

    Args:
        closes: Series of close prices.
        window: Moving average period.
        std_dev: Number of standard deviations for bands.

    Returns:
        DataFrame with columns: upper, middle, lower.
    """
    bb = BollingerBands(close=closes, window=window, window_dev=std_dev)
    return pd.DataFrame(
        {
            "upper": bb.bollinger_hband(),
            "middle": bb.bollinger_mavg(),
            "lower": bb.bollinger_lband(),
        }
    )


def compute_all_indicators(closes: pd.Series) -> pd.DataFrame:
    """Calculate all standard technical indicators for a ticker.

    Args:
        closes: Series of close prices indexed by date.

    Returns:
        DataFrame with all indicator columns, indexed by date.
    """
    rsi = compute_rsi(closes)
    macd_df = compute_macd(closes)
    sma_20 = compute_sma(closes, window=20)
    sma_50 = compute_sma(closes, window=50)
    sma_200 = compute_sma(closes, window=200)
    bb = compute_bollinger_bands(closes)

    result = pd.DataFrame(
        {
            "close": closes,
            "rsi_14": rsi,
            "macd": macd_df["macd"],
            "macd_signal": macd_df["signal"],
            "macd_hist": macd_df["histogram"],
            "sma_20": sma_20,
            "sma_50": sma_50,
            "sma_200": sma_200,
            "bb_upper": bb["upper"],
            "bb_middle": bb["middle"],
            "bb_lower": bb["lower"],
        }
    )

    log.debug(
        "indicators_computed",
        rows=len(result),
        latest_rsi=round(rsi.iloc[-1], 2) if not rsi.empty else None,
    )

    return result
