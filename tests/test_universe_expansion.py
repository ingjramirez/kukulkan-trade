"""Tests for Phase 19 universe expansion (28 new tickers)."""

from config.risk_rules import RISK_RULES
from config.universe import (
    FULL_UNIVERSE,
    PORTFOLIO_A_UNIVERSE,
    PORTFOLIO_B_UNIVERSE,
    SECTOR_ETFS,
    SECTOR_MAP,
    THEMATIC_ETFS,
)
from src.agent.strategy_directives import CONSERVATIVE_DIRECTIVE


# All 28 tickers from Phase 19 (including 5 that already existed)
PHASE_19_TICKERS = [
    # Fixed Income
    "BIL", "SHY", "IEF", "AGG", "VTIP", "HYG",
    # International — Broad
    "VEA", "VWO", "EFA", "IXUS",
    # International — Regional
    "FXI", "KWEB", "INDA", "VGK",
    # Real Estate
    "VNQ", "SCHH",
    # Dividend / Value
    "SCHD", "VTV", "DVY",
    # Commodities
    "DBC", "PDBC",
    # Individual Stocks
    "AVGO", "UNH", "PG", "XOM", "KO",
    # Thematic
    "ICLN",
    # Volatility Hedge
    "VIXY",
]


def test_new_tickers_in_portfolio_b():
    """All 28 Phase 19 tickers must be present in PORTFOLIO_B_UNIVERSE."""
    missing = [t for t in PHASE_19_TICKERS if t not in PORTFOLIO_B_UNIVERSE]
    assert missing == [], f"Missing from PORTFOLIO_B_UNIVERSE: {missing}"


def test_portfolio_a_unchanged():
    """Portfolio A should still be exactly SECTOR_ETFS + THEMATIC_ETFS (20 tickers)."""
    expected = SECTOR_ETFS + THEMATIC_ETFS
    assert PORTFOLIO_A_UNIVERSE == expected
    assert len(PORTFOLIO_A_UNIVERSE) == 20


def test_sector_map_complete():
    """Every ticker in FULL_UNIVERSE must have a SECTOR_MAP entry."""
    missing = [t for t in FULL_UNIVERSE if t not in SECTOR_MAP]
    assert missing == [], f"Missing SECTOR_MAP entries: {missing}"


def test_no_duplicate_tickers():
    """FULL_UNIVERSE should have no duplicates."""
    assert len(FULL_UNIVERSE) == len(set(FULL_UNIVERSE))


def test_risk_sector_overrides():
    """Per-sector concentration limits should be correct."""
    overrides = RISK_RULES.sector_concentration_overrides
    assert overrides["Fixed Income"] == 0.50
    assert overrides["Hedge"] == 0.05
    assert overrides["Crypto"] == 0.05
    assert overrides["International"] == 0.25
    # Unknown sector falls back to global limit
    assert overrides.get("SomeFakeSector", RISK_RULES.max_sector_concentration) == 0.50


def test_defensive_tickers_include_bonds():
    """New bond ETFs should be in defensive_tickers."""
    for bond in ("BIL", "SHY", "IEF", "AGG", "VTIP"):
        assert bond in RISK_RULES.defensive_tickers, f"{bond} missing from defensive_tickers"


def test_conservative_directive_mentions_new_instruments():
    """CONSERVATIVE_DIRECTIVE should reference key new instruments."""
    for ticker in ("BIL", "VEA", "FXI", "SCHD", "VIXY"):
        assert ticker in CONSERVATIVE_DIRECTIVE, (
            f"{ticker} not mentioned in CONSERVATIVE_DIRECTIVE"
        )
