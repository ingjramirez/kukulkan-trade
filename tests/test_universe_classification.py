"""Tests for instrument classification in config/universe.py."""

from config.universe import (
    INVERSE_ETF_META,
    INVERSE_ETFS,
    PORTFOLIO_B_UNIVERSE,
    SECTOR_MAP,
    InstrumentType,
    classify_instrument,
    is_equity_hedge,
)


class TestInstrumentType:
    def test_enum_values(self) -> None:
        assert InstrumentType.STOCK.value == "stock"
        assert InstrumentType.ETF.value == "etf"
        assert InstrumentType.INVERSE_ETF.value == "inverse_etf"
        assert InstrumentType.CRYPTO_PROXY.value == "crypto_proxy"


class TestClassifyInstrument:
    def test_stocks(self) -> None:
        for ticker in ("AAPL", "MSFT", "GOOGL", "NVDA", "TSLA"):
            assert classify_instrument(ticker) == InstrumentType.STOCK

    def test_additional_stocks(self) -> None:
        assert classify_instrument("AVGO") == InstrumentType.STOCK
        assert classify_instrument("KO") == InstrumentType.STOCK

    def test_etfs(self) -> None:
        for ticker in ("XLK", "QQQ", "SPY", "GLD", "TLT"):
            assert classify_instrument(ticker) == InstrumentType.ETF

    def test_inverse_etfs(self) -> None:
        for ticker in ("SH", "PSQ", "RWM", "TBF"):
            assert classify_instrument(ticker) == InstrumentType.INVERSE_ETF

    def test_crypto_proxy(self) -> None:
        assert classify_instrument("IBIT") == InstrumentType.CRYPTO_PROXY

    def test_unknown_ticker_defaults_to_etf(self) -> None:
        # Unknown tickers that aren't in any specific list default to ETF
        assert classify_instrument("ZZZZZ") == InstrumentType.ETF


class TestIsEquityHedge:
    def test_equity_hedges_true(self) -> None:
        assert is_equity_hedge("SH") is True
        assert is_equity_hedge("PSQ") is True
        assert is_equity_hedge("RWM") is True

    def test_tbf_not_equity_hedge(self) -> None:
        assert is_equity_hedge("TBF") is False

    def test_non_inverse_returns_false(self) -> None:
        assert is_equity_hedge("AAPL") is False
        assert is_equity_hedge("XLK") is False
        assert is_equity_hedge("SPY") is False


class TestInverseEtfMeta:
    def test_has_all_four_tickers(self) -> None:
        assert set(INVERSE_ETF_META.keys()) == {"SH", "PSQ", "RWM", "TBF"}

    def test_required_fields(self) -> None:
        for ticker, meta in INVERSE_ETF_META.items():
            assert "benchmark" in meta, f"{ticker} missing benchmark"
            assert "leverage" in meta, f"{ticker} missing leverage"
            assert "description" in meta, f"{ticker} missing description"
            assert "equity_hedge" in meta, f"{ticker} missing equity_hedge"

    def test_no_leveraged_etfs(self) -> None:
        for ticker, meta in INVERSE_ETF_META.items():
            assert meta["leverage"] == 1, f"{ticker} has leverage != 1"

    def test_benchmarks_valid(self) -> None:
        valid_benchmarks = {"SPY", "QQQ", "IWM", "TLT"}
        for ticker, meta in INVERSE_ETF_META.items():
            assert meta["benchmark"] in valid_benchmarks, f"{ticker} has invalid benchmark"


class TestBackwardCompat:
    def test_inverse_etfs_list(self) -> None:
        assert isinstance(INVERSE_ETFS, list)
        assert set(INVERSE_ETFS) == set(INVERSE_ETF_META.keys())

    def test_rwm_in_portfolio_b_universe(self) -> None:
        assert "RWM" in PORTFOLIO_B_UNIVERSE

    def test_rwm_in_sector_map(self) -> None:
        assert SECTOR_MAP.get("RWM") == "Inverse"

    def test_all_inverse_etfs_in_sector_map(self) -> None:
        for ticker in INVERSE_ETFS:
            assert ticker in SECTOR_MAP, f"{ticker} not in SECTOR_MAP"
            assert SECTOR_MAP[ticker] == "Inverse"
