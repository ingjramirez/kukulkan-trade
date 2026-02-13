"""Tests for momentum calculator."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from src.analysis.momentum import calculate_momentum, get_top_n, momentum_to_db_rows


@pytest.fixture
def sample_closes() -> pd.DataFrame:
    """Create synthetic close prices for 5 tickers over 80 trading days."""
    np.random.seed(42)
    dates = pd.bdate_range(end="2026-02-05", periods=80)

    # Create prices with different momentum profiles
    data = {
        "XLK": 200 + np.cumsum(np.random.normal(0.5, 1, 80)),  # strong uptrend
        "XLF": 40 + np.cumsum(np.random.normal(0.2, 0.5, 80)),  # moderate up
        "XLE": 80 + np.cumsum(np.random.normal(-0.1, 1, 80)),  # flat/down
        "XLV": 150 + np.cumsum(np.random.normal(0.3, 0.8, 80)),  # moderate up
        "XLU": 70 + np.cumsum(np.random.normal(-0.3, 0.5, 80)),  # downtrend
    }
    return pd.DataFrame(data, index=dates)


class TestCalculateMomentum:
    def test_returns_all_tickers(self, sample_closes: pd.DataFrame) -> None:
        result = calculate_momentum(sample_closes)
        assert len(result) == 5
        assert set(result["ticker"]) == {"XLK", "XLF", "XLE", "XLV", "XLU"}

    def test_ranks_are_sequential(self, sample_closes: pd.DataFrame) -> None:
        result = calculate_momentum(sample_closes)
        assert list(result["rank"]) == [1, 2, 3, 4, 5]

    def test_rank_1_has_highest_return(self, sample_closes: pd.DataFrame) -> None:
        result = calculate_momentum(sample_closes)
        returns = result["return_63d"].values
        assert returns[0] == max(returns)

    def test_returns_are_floats(self, sample_closes: pd.DataFrame) -> None:
        result = calculate_momentum(sample_closes)
        assert result["return_63d"].dtype in [np.float64, np.float32]

    def test_insufficient_data_returns_empty(self) -> None:
        short_data = pd.DataFrame(
            {"XLK": [100, 101, 102]},
            index=pd.bdate_range(end="2026-02-05", periods=3),
        )
        result = calculate_momentum(short_data)
        assert len(result) == 0

    def test_skip_parameter_works(self, sample_closes: pd.DataFrame) -> None:
        result_skip5 = calculate_momentum(sample_closes, skip=5)
        result_skip0 = calculate_momentum(sample_closes, skip=0)
        # Different skip values should generally produce different returns
        assert not result_skip5["return_63d"].equals(result_skip0["return_63d"])

    def test_custom_lookback(self, sample_closes: pd.DataFrame) -> None:
        result = calculate_momentum(sample_closes, lookback=20, skip=3)
        assert len(result) == 5

    def test_handles_nan_tickers(self) -> None:
        dates = pd.bdate_range(end="2026-02-05", periods=80)
        data = {
            "XLK": 200 + np.cumsum(np.random.normal(0.5, 1, 80)),
            "BAD": [np.nan] * 80,
        }
        df = pd.DataFrame(data, index=dates)
        result = calculate_momentum(df)
        assert "BAD" not in result["ticker"].values


class TestGetTopN:
    def test_top_1(self, sample_closes: pd.DataFrame) -> None:
        rankings = calculate_momentum(sample_closes)
        top = get_top_n(rankings, n=1)
        assert len(top) == 1
        assert top[0] == rankings.iloc[0]["ticker"]

    def test_top_3(self, sample_closes: pd.DataFrame) -> None:
        rankings = calculate_momentum(sample_closes)
        top = get_top_n(rankings, n=3)
        assert len(top) == 3


class TestMomentumToDbRows:
    def test_conversion(self, sample_closes: pd.DataFrame) -> None:
        rankings = calculate_momentum(sample_closes)
        rows = momentum_to_db_rows(rankings, date(2026, 2, 5))
        assert len(rows) == 5
        assert rows[0].date == date(2026, 2, 5)
        assert rows[0].rank == 1
