"""Tests for crypto asset support in universe, classification, and ticker mapping."""

from config.universe import (
    CRYPTO_TICKERS,
    FULL_UNIVERSE,
    PORTFOLIO_B_UNIVERSE,
    SECTOR_MAP,
    InstrumentType,
    classify_instrument,
)
from src.utils.ticker_mapping import (
    ALPACA_TO_CANONICAL,
    CRYPTO_TICKER_MAP,
    is_crypto_ticker,
    to_alpaca_format,
    to_canonical_format,
)


class TestCryptoUniverse:
    def test_btc_in_crypto_tickers(self) -> None:
        assert "BTC-USD" in CRYPTO_TICKERS

    def test_btc_in_full_universe(self) -> None:
        assert "BTC-USD" in FULL_UNIVERSE

    def test_btc_in_portfolio_b_universe(self) -> None:
        assert "BTC-USD" in PORTFOLIO_B_UNIVERSE

    def test_ibit_still_in_universe(self) -> None:
        assert "IBIT" in FULL_UNIVERSE
        assert "IBIT" in PORTFOLIO_B_UNIVERSE


class TestCryptoClassification:
    def test_classify_btc_is_crypto(self) -> None:
        assert classify_instrument("BTC-USD") == InstrumentType.CRYPTO

    def test_classify_ibit_still_crypto_proxy(self) -> None:
        assert classify_instrument("IBIT") == InstrumentType.CRYPTO_PROXY

    def test_crypto_enum_value(self) -> None:
        assert InstrumentType.CRYPTO.value == "crypto"

    def test_classify_stocks_unaffected(self) -> None:
        assert classify_instrument("AAPL") == InstrumentType.STOCK

    def test_classify_etfs_unaffected(self) -> None:
        assert classify_instrument("XLK") == InstrumentType.ETF

    def test_classify_inverse_unaffected(self) -> None:
        assert classify_instrument("SH") == InstrumentType.INVERSE_ETF


class TestCryptoSectorMap:
    def test_btc_sector_is_crypto(self) -> None:
        assert SECTOR_MAP["BTC-USD"] == "Crypto"

    def test_ibit_sector_still_crypto(self) -> None:
        assert SECTOR_MAP["IBIT"] == "Crypto"


class TestTickerMapping:
    def test_to_alpaca_format_btc(self) -> None:
        assert to_alpaca_format("BTC-USD") == "BTC/USD"

    def test_to_alpaca_format_equity_unchanged(self) -> None:
        assert to_alpaca_format("AAPL") == "AAPL"
        assert to_alpaca_format("XLK") == "XLK"

    def test_to_canonical_format_btc(self) -> None:
        assert to_canonical_format("BTC/USD") == "BTC-USD"

    def test_to_canonical_format_equity_unchanged(self) -> None:
        assert to_canonical_format("AAPL") == "AAPL"

    def test_is_crypto_ticker_canonical(self) -> None:
        assert is_crypto_ticker("BTC-USD") is True

    def test_is_crypto_ticker_alpaca(self) -> None:
        assert is_crypto_ticker("BTC/USD") is True

    def test_is_crypto_ticker_equity(self) -> None:
        assert is_crypto_ticker("AAPL") is False
        assert is_crypto_ticker("IBIT") is False

    def test_roundtrip_mapping(self) -> None:
        for canonical, alpaca in CRYPTO_TICKER_MAP.items():
            assert to_canonical_format(to_alpaca_format(canonical)) == canonical
            assert to_alpaca_format(to_canonical_format(alpaca)) == alpaca

    def test_alpaca_to_canonical_reverse_map(self) -> None:
        assert ALPACA_TO_CANONICAL == {"BTC/USD": "BTC-USD"}
