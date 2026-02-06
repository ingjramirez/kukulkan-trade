"""Tests for the dashboard data loading functions.

Uses a sync in-memory SQLite database to verify data queries work correctly.
Streamlit UI rendering is not tested here (requires browser/Selenium).
"""

from datetime import date, datetime

import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from src.storage.models import (
    AgentDecisionRow,
    Base,
    DailySnapshotRow,
    MomentumRankingRow,
    PortfolioRow,
    PositionRow,
    TradeRow,
)


@pytest.fixture
def sync_session():
    """Create a sync in-memory SQLite database for testing."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    factory = sessionmaker(engine, expire_on_commit=False)
    with factory() as session:
        yield session


@pytest.fixture
def seeded_session(sync_session: Session):
    """Session with sample data for all tables."""
    s = sync_session

    # Portfolios
    for name, val in [("A", 34000), ("B", 67000)]:
        s.add(PortfolioRow(name=name, cash=5000, total_value=val))

    # Positions
    s.add(PositionRow(portfolio="A", ticker="QQQ", shares=10, avg_price=500))
    s.add(PositionRow(portfolio="B", ticker="XLK", shares=20, avg_price=200))
    s.add(PositionRow(portfolio="B", ticker="XLF", shares=30, avg_price=40))

    # Snapshots (3 days x 2 portfolios)
    for day_offset in range(3):
        d = date(2026, 2, 3 + day_offset)
        for name, base_val in [("A", 33000), ("B", 66000)]:
            val = base_val + day_offset * 200 + (ord(name) - 64) * 100
            s.add(DailySnapshotRow(
                portfolio=name,
                date=d,
                total_value=val,
                cash=5000,
                positions_value=val - 5000,
                daily_return_pct=0.5 * day_offset,
                cumulative_return_pct=0.5 * day_offset,
            ))

    # Trades
    s.add(TradeRow(portfolio="A", ticker="QQQ", side="BUY", shares=10, price=500, total=5000, reason="momentum"))
    s.add(TradeRow(portfolio="B", ticker="XLK", side="BUY", shares=20, price=200, total=4000))
    s.add(TradeRow(portfolio="B", ticker="XLF", side="SELL", shares=5, price=42, total=210, reason="rotation"))

    # Momentum rankings
    for i, ticker in enumerate(["QQQ", "SMH", "XLK", "IWM", "EFA"]):
        s.add(MomentumRankingRow(date=date(2026, 2, 5), ticker=ticker, return_63d=15.0 - i * 2, rank=i + 1))

    # Agent decisions
    s.add(AgentDecisionRow(
        date=date(2026, 2, 5),
        reasoning="Bullish outlook, buying tech",
        proposed_trades='[{"ticker": "XLK", "side": "BUY", "weight": 0.15}]',
        model_used="claude-sonnet-4-5-20250929",
        tokens_used=200,
    ))

    s.commit()
    return s


# ── Query helpers (mirror dashboard logic without Streamlit caching) ─────────

def _load_snapshots(session: Session) -> pd.DataFrame:
    from sqlalchemy import select
    rows = session.execute(
        select(DailySnapshotRow).order_by(DailySnapshotRow.date)
    ).scalars().all()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame([{
        "portfolio": r.portfolio, "date": r.date, "total_value": r.total_value,
        "cash": r.cash, "positions_value": r.positions_value,
        "daily_return_pct": r.daily_return_pct,
    } for r in rows])


def _load_portfolios(session: Session) -> dict:
    from sqlalchemy import select
    rows = session.execute(select(PortfolioRow)).scalars().all()
    return {r.name: {"cash": r.cash, "total_value": r.total_value} for r in rows}


def _load_positions(session: Session) -> pd.DataFrame:
    from sqlalchemy import select
    rows = session.execute(select(PositionRow)).scalars().all()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame([{
        "portfolio": r.portfolio, "ticker": r.ticker, "shares": r.shares,
        "avg_price": r.avg_price,
    } for r in rows])


def _load_trades(session: Session) -> pd.DataFrame:
    from sqlalchemy import select
    rows = session.execute(
        select(TradeRow).order_by(TradeRow.executed_at.desc())
    ).scalars().all()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame([{
        "portfolio": r.portfolio, "ticker": r.ticker, "side": r.side,
        "shares": r.shares, "price": r.price, "total": r.total, "reason": r.reason or "",
    } for r in rows])


def _load_momentum(session: Session) -> pd.DataFrame:
    from sqlalchemy import select
    latest = session.execute(
        select(MomentumRankingRow.date).order_by(MomentumRankingRow.date.desc()).limit(1)
    ).scalar_one_or_none()
    if not latest:
        return pd.DataFrame()
    rows = session.execute(
        select(MomentumRankingRow).where(MomentumRankingRow.date == latest)
        .order_by(MomentumRankingRow.rank)
    ).scalars().all()
    return pd.DataFrame([{"ticker": r.ticker, "return_63d": r.return_63d, "rank": r.rank} for r in rows])


def _load_decisions(session: Session) -> pd.DataFrame:
    from sqlalchemy import select
    rows = session.execute(
        select(AgentDecisionRow).order_by(AgentDecisionRow.date.desc()).limit(30)
    ).scalars().all()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame([{
        "date": r.date, "reasoning": r.reasoning, "tokens": r.tokens_used,
    } for r in rows])


# ── Tests ────────────────────────────────────────────────────────────────────


class TestLoadPortfolios:
    def test_returns_both(self, seeded_session: Session) -> None:
        portfolios = _load_portfolios(seeded_session)
        assert set(portfolios.keys()) == {"A", "B"}

    def test_values_correct(self, seeded_session: Session) -> None:
        portfolios = _load_portfolios(seeded_session)
        assert portfolios["A"]["total_value"] == 34000
        assert portfolios["B"]["cash"] == 5000

    def test_empty_db(self, sync_session: Session) -> None:
        portfolios = _load_portfolios(sync_session)
        assert portfolios == {}


class TestLoadSnapshots:
    def test_returns_dataframe(self, seeded_session: Session) -> None:
        df = _load_snapshots(seeded_session)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 6  # 3 days x 2 portfolios

    def test_sorted_by_date(self, seeded_session: Session) -> None:
        df = _load_snapshots(seeded_session)
        dates = df["date"].tolist()
        assert dates == sorted(dates)

    def test_empty_db(self, sync_session: Session) -> None:
        df = _load_snapshots(sync_session)
        assert df.empty


class TestLoadPositions:
    def test_returns_all_positions(self, seeded_session: Session) -> None:
        df = _load_positions(seeded_session)
        assert len(df) == 3  # QQQ + XLK + XLF

    def test_filter_by_portfolio(self, seeded_session: Session) -> None:
        df = _load_positions(seeded_session)
        assert len(df[df["portfolio"] == "B"]) == 2

    def test_empty_db(self, sync_session: Session) -> None:
        df = _load_positions(sync_session)
        assert df.empty


class TestLoadTrades:
    def test_returns_all_trades(self, seeded_session: Session) -> None:
        df = _load_trades(seeded_session)
        assert len(df) == 3

    def test_buy_sell_sides(self, seeded_session: Session) -> None:
        df = _load_trades(seeded_session)
        assert "BUY" in df["side"].values
        assert "SELL" in df["side"].values

    def test_totals_correct(self, seeded_session: Session) -> None:
        df = _load_trades(seeded_session)
        qqq = df[df["ticker"] == "QQQ"].iloc[0]
        assert qqq["total"] == 5000

    def test_empty_db(self, sync_session: Session) -> None:
        df = _load_trades(sync_session)
        assert df.empty


class TestLoadMomentum:
    def test_returns_rankings(self, seeded_session: Session) -> None:
        df = _load_momentum(seeded_session)
        assert len(df) == 5

    def test_sorted_by_rank(self, seeded_session: Session) -> None:
        df = _load_momentum(seeded_session)
        assert df["rank"].tolist() == [1, 2, 3, 4, 5]

    def test_top_ticker(self, seeded_session: Session) -> None:
        df = _load_momentum(seeded_session)
        assert df.iloc[0]["ticker"] == "QQQ"

    def test_empty_db(self, sync_session: Session) -> None:
        df = _load_momentum(sync_session)
        assert df.empty


class TestLoadDecisions:
    def test_returns_decisions(self, seeded_session: Session) -> None:
        df = _load_decisions(seeded_session)
        assert len(df) == 1

    def test_reasoning_present(self, seeded_session: Session) -> None:
        df = _load_decisions(seeded_session)
        assert "Bullish" in df.iloc[0]["reasoning"]

    def test_tokens_tracked(self, seeded_session: Session) -> None:
        df = _load_decisions(seeded_session)
        assert df.iloc[0]["tokens"] == 200

    def test_empty_db(self, sync_session: Session) -> None:
        df = _load_decisions(sync_session)
        assert df.empty
