"""Macro economic data fetcher using FRED API.

Fetches key indicators for Portfolio B regime detection:
- Yield curve (10Y-2Y Treasury spread)
- VIX (volatility index)
- Initial jobless claims
- Fed funds rate
"""

from datetime import date, timedelta

import pandas as pd
import structlog
from fredapi import Fred

from config.settings import settings
from src.storage.database import Database
from src.storage.models import MacroDataRow
from src.utils.retry import retry_macro_data

log = structlog.get_logger()

# FRED series IDs for key macro indicators
FRED_SERIES: dict[str, str] = {
    "yield_curve_10y2y": "T10Y2Y",  # 10-Year minus 2-Year Treasury spread
    "vix": "VIXCLS",  # CBOE Volatility Index
    "initial_claims": "ICSA",  # Initial jobless claims (weekly)
    "fed_funds_rate": "FEDFUNDS",  # Effective federal funds rate
    "unemployment_rate": "UNRATE",  # Unemployment rate
    "cpi_yoy": "CPIAUCSL",  # CPI (for year-over-year inflation calc)
}


class MacroDataFetcher:
    """Fetches and caches macro indicators from FRED."""

    def __init__(self, db: Database, api_key: str | None = None) -> None:
        self._db = db
        self._api_key = api_key or settings.fred_api_key
        self._fred: Fred | None = None

    @property
    def fred(self) -> Fred:
        """Lazy-init FRED client."""
        if self._fred is None:
            if not self._api_key:
                raise ValueError("FRED_API_KEY not set. Get one at https://fred.stlouisfed.org")
            self._fred = Fred(api_key=self._api_key)
        return self._fred

    @retry_macro_data
    def fetch_series(
        self,
        series_id: str,
        start: date | None = None,
        end: date | None = None,
    ) -> pd.Series:
        """Fetch a single FRED series.

        Args:
            series_id: FRED series ID (e.g., 'T10Y2Y').
            start: Start date (defaults to 1 year ago).
            end: End date (defaults to today).

        Returns:
            pandas Series indexed by date.
        """
        start = start or date.today() - timedelta(days=365)
        end = end or date.today()

        log.debug("fetching_fred_series", series_id=series_id)
        data = self.fred.get_series(
            series_id,
            observation_start=start.isoformat(),
            observation_end=end.isoformat(),
        )
        return data.dropna()

    def fetch_all_indicators(
        self,
        start: date | None = None,
        end: date | None = None,
    ) -> dict[str, pd.Series]:
        """Fetch all configured macro indicators.

        Args:
            start: Start date.
            end: End date.

        Returns:
            Dict mapping indicator name -> Series.
        """
        results: dict[str, pd.Series] = {}
        for name, series_id in FRED_SERIES.items():
            try:
                results[name] = self.fetch_series(series_id, start=start, end=end)
                log.debug("fred_series_fetched", name=name, points=len(results[name]))
            except Exception:
                log.warning("fred_series_failed", name=name, series_id=series_id)
        log.info("macro_data_fetched", indicators=len(results))
        return results

    async def fetch_and_cache(
        self,
        start: date | None = None,
        end: date | None = None,
    ) -> dict[str, pd.Series]:
        """Fetch all indicators and persist to SQLite.

        Args:
            start: Start date.
            end: End date.

        Returns:
            Dict mapping indicator name -> Series.
        """
        data = self.fetch_all_indicators(start=start, end=end)

        async with self._db.session() as s:
            for name, series in data.items():
                for dt, value in series.items():
                    row_date = dt.date() if hasattr(dt, "date") else dt
                    s.add(
                        MacroDataRow(
                            indicator=name,
                            date=row_date,
                            value=float(value),
                        )
                    )
            await s.commit()

        log.info("macro_data_cached", indicators=len(data))
        return data

    def get_latest_yield_curve(self) -> float | None:
        """Get the most recent 10Y-2Y spread value.

        Returns:
            Latest spread value, or None if unavailable.
            Negative = inverted yield curve (recession signal).
        """
        try:
            series = self.fetch_series("T10Y2Y")
            if not series.empty:
                return float(series.iloc[-1])
        except Exception:
            log.warning("yield_curve_fetch_failed")
        return None

    def get_latest_vix(self) -> float | None:
        """Get the most recent VIX value.

        Returns:
            Latest VIX value, or None.
            >30 = high fear, <15 = complacency.
        """
        try:
            series = self.fetch_series("VIXCLS")
            if not series.empty:
                return float(series.iloc[-1])
        except Exception:
            log.warning("vix_fetch_failed")
        return None
