"""Streamlit dashboard for portfolio visualization.

Shows portfolio performance, positions, trade history,
and strategy comparisons across both portfolios.

Usage:
    streamlit run src/dashboard/app.py
"""

import json
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import create_engine, select, text
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

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Atlas Trading Bot",
    page_icon="📊",
    layout="wide",
)

# ── Database connection ──────────────────────────────────────────────────────

DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "atlas.db"


@st.cache_resource
def get_engine():
    """Create a sync SQLAlchemy engine (cached across reruns)."""
    url = f"sqlite:///{DB_PATH}"
    return create_engine(url, echo=False)


def get_session() -> Session:
    """Create a new sync session."""
    engine = get_engine()
    factory = sessionmaker(engine, expire_on_commit=False)
    return factory()


# ── Data loading ─────────────────────────────────────────────────────────────


@st.cache_data(ttl=60)
def load_snapshots() -> pd.DataFrame:
    """Load all daily snapshots into a DataFrame."""
    with get_session() as s:
        rows = s.execute(
            select(DailySnapshotRow).order_by(DailySnapshotRow.date)
        ).scalars().all()

    if not rows:
        return pd.DataFrame()

    data = [
        {
            "portfolio": r.portfolio,
            "date": r.date,
            "total_value": r.total_value,
            "cash": r.cash,
            "positions_value": r.positions_value,
            "daily_return_pct": r.daily_return_pct,
            "cumulative_return_pct": r.cumulative_return_pct,
        }
        for r in rows
    ]
    return pd.DataFrame(data)


@st.cache_data(ttl=60)
def load_portfolios() -> dict[str, dict]:
    """Load current portfolio states."""
    with get_session() as s:
        rows = s.execute(select(PortfolioRow)).scalars().all()

    return {
        r.name: {
            "cash": r.cash,
            "total_value": r.total_value,
            "updated_at": r.updated_at,
        }
        for r in rows
    }


@st.cache_data(ttl=60)
def load_positions() -> pd.DataFrame:
    """Load all open positions."""
    with get_session() as s:
        rows = s.execute(select(PositionRow)).scalars().all()

    if not rows:
        return pd.DataFrame()

    data = [
        {
            "portfolio": r.portfolio,
            "ticker": r.ticker,
            "shares": r.shares,
            "avg_price": r.avg_price,
            "market_value": r.market_value if r.market_value is not None else r.shares * r.avg_price,
        }
        for r in rows
    ]
    return pd.DataFrame(data)


@st.cache_data(ttl=60)
def load_trades() -> pd.DataFrame:
    """Load all trades."""
    with get_session() as s:
        rows = s.execute(
            select(TradeRow).order_by(TradeRow.executed_at.desc())
        ).scalars().all()

    if not rows:
        return pd.DataFrame()

    data = [
        {
            "portfolio": r.portfolio,
            "ticker": r.ticker,
            "side": r.side,
            "shares": r.shares,
            "price": r.price,
            "total": r.total,
            "reason": r.reason or "",
            "executed_at": r.executed_at,
        }
        for r in rows
    ]
    return pd.DataFrame(data)


@st.cache_data(ttl=60)
def load_momentum_rankings() -> pd.DataFrame:
    """Load latest momentum rankings."""
    with get_session() as s:
        # Find latest date
        latest = s.execute(
            select(MomentumRankingRow.date)
            .order_by(MomentumRankingRow.date.desc())
            .limit(1)
        ).scalar_one_or_none()

        if not latest:
            return pd.DataFrame()

        rows = s.execute(
            select(MomentumRankingRow)
            .where(MomentumRankingRow.date == latest)
            .order_by(MomentumRankingRow.rank)
        ).scalars().all()

    data = [
        {"ticker": r.ticker, "return_63d": r.return_63d, "rank": r.rank}
        for r in rows
    ]
    return pd.DataFrame(data)


@st.cache_data(ttl=60)
def load_agent_decisions() -> pd.DataFrame:
    """Load Claude agent decisions for Portfolio B."""
    with get_session() as s:
        rows = s.execute(
            select(AgentDecisionRow)
            .order_by(AgentDecisionRow.date.desc())
            .limit(30)
        ).scalars().all()

    if not rows:
        return pd.DataFrame()

    data = [
        {
            "date": r.date,
            "reasoning": r.reasoning or "",
            "proposed_trades": r.proposed_trades or "[]",
            "model": r.model_used or "",
            "tokens": r.tokens_used or 0,
        }
        for r in rows
    ]
    return pd.DataFrame(data)


# ── Sidebar ──────────────────────────────────────────────────────────────────

st.sidebar.title("Atlas Trading Bot")
page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Portfolio A", "Portfolio B", "Trade Log"],
)


# ── Helper functions ─────────────────────────────────────────────────────────

STRATEGY_LABELS = {
    "A": "Momentum",
    "B": "AI Autonomy",
}

COLORS = {
    "A": "#636EFA",   # blue
    "B": "#00CC96",   # green
}

INITIAL_VALUES = {
    "A": 33_000.0,
    "B": 66_000.0,
}


def no_data_warning() -> None:
    """Show a warning when the bot hasn't run yet."""
    st.warning(
        "No data yet. Run the pipeline first:\n\n"
        "```bash\npython -m src.main --run-now\n```"
    )


# ── Overview page ────────────────────────────────────────────────────────────


def page_overview() -> None:
    """Main overview with combined equity curve and portfolio summary."""
    st.title("Portfolio Overview")

    portfolios = load_portfolios()
    snapshots = load_snapshots()

    if not portfolios:
        no_data_warning()
        return

    # ── KPI metrics row ──────────────────────────────────────────────
    cols = st.columns(3)

    total_value = sum(p["total_value"] for p in portfolios.values())
    initial = 99_000.0
    total_return = ((total_value - initial) / initial) * 100

    cols[0].metric("Combined Value", f"${total_value:,.0f}", f"{total_return:+.2f}%")

    for i, name in enumerate(("A", "B")):
        if name in portfolios:
            p = portfolios[name]
            init_val = INITIAL_VALUES[name]
            ret = ((p["total_value"] - init_val) / init_val) * 100
            cols[i + 1].metric(
                f"Portfolio {name} ({STRATEGY_LABELS[name]})",
                f"${p['total_value']:,.0f}",
                f"{ret:+.2f}%",
            )

    # ── Equity curve ─────────────────────────────────────────────────
    if not snapshots.empty:
        st.subheader("Equity Curves")

        # Per-portfolio lines
        fig = px.line(
            snapshots,
            x="date",
            y="total_value",
            color="portfolio",
            color_discrete_map=COLORS,
            labels={"total_value": "Value ($)", "date": "Date", "portfolio": "Portfolio"},
        )

        # Combined line
        combined = snapshots.groupby("date")["total_value"].sum().reset_index()
        fig.add_trace(
            go.Scatter(
                x=combined["date"],
                y=combined["total_value"],
                name="Combined",
                line=dict(color="#FFA15A", width=3, dash="dash"),
            )
        )

        fig.update_layout(
            height=450,
            hovermode="x unified",
            yaxis_tickprefix="$",
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Daily returns ────────────────────────────────────────────
        st.subheader("Daily Returns")

        returns_df = snapshots[snapshots["daily_return_pct"].notna()].copy()
        if not returns_df.empty:
            fig2 = px.bar(
                returns_df,
                x="date",
                y="daily_return_pct",
                color="portfolio",
                color_discrete_map=COLORS,
                barmode="group",
                labels={"daily_return_pct": "Return (%)", "date": "Date"},
            )
            fig2.update_layout(height=350, hovermode="x unified")
            st.plotly_chart(fig2, use_container_width=True)

    # ── Current positions ────────────────────────────────────────────
    positions = load_positions()
    if not positions.empty:
        st.subheader("Current Positions")
        st.dataframe(
            positions.style.format({
                "shares": "{:.0f}",
                "avg_price": "${:.2f}",
                "market_value": "${:,.0f}",
            }),
            use_container_width=True,
            hide_index=True,
        )


# ── Portfolio detail pages ───────────────────────────────────────────────────


def page_portfolio(name: str) -> None:
    """Detail page for a single portfolio."""
    label = STRATEGY_LABELS[name]
    st.title(f"Portfolio {name} — {label}")

    portfolios = load_portfolios()
    snapshots = load_snapshots()

    if name not in portfolios:
        no_data_warning()
        return

    p = portfolios[name]
    init_val = INITIAL_VALUES[name]
    ret = ((p["total_value"] - init_val) / init_val) * 100

    # ── KPI row ──────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Value", f"${p['total_value']:,.0f}", f"{ret:+.2f}%")
    c2.metric("Cash", f"${p['cash']:,.0f}")
    c3.metric("Last Updated", p["updated_at"].strftime("%Y-%m-%d %H:%M") if p["updated_at"] else "—")

    # ── Equity curve ─────────────────────────────────────────────────
    port_snaps = snapshots[snapshots["portfolio"] == name] if not snapshots.empty else pd.DataFrame()
    if not port_snaps.empty:
        st.subheader("Equity Curve")
        fig = px.area(
            port_snaps,
            x="date",
            y="total_value",
            labels={"total_value": "Value ($)", "date": "Date"},
            color_discrete_sequence=[COLORS[name]],
        )
        fig.update_layout(height=400, yaxis_tickprefix="$")
        st.plotly_chart(fig, use_container_width=True)

        # Cash vs invested
        st.subheader("Cash vs Invested")
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=port_snaps["date"], y=port_snaps["cash"], name="Cash"))
        fig2.add_trace(go.Bar(x=port_snaps["date"], y=port_snaps["positions_value"], name="Invested"))
        fig2.update_layout(barmode="stack", height=350, yaxis_tickprefix="$")
        st.plotly_chart(fig2, use_container_width=True)

    # ── Positions ────────────────────────────────────────────────────
    positions = load_positions()
    port_pos = positions[positions["portfolio"] == name] if not positions.empty else pd.DataFrame()
    if not port_pos.empty:
        st.subheader("Open Positions")
        st.dataframe(
            port_pos[["ticker", "shares", "avg_price", "market_value"]].style.format({
                "shares": "{:.0f}",
                "avg_price": "${:.2f}",
                "market_value": "${:,.0f}",
            }),
            use_container_width=True,
            hide_index=True,
        )

    # ── Strategy-specific sections ───────────────────────────────────
    if name == "A":
        _section_momentum()
    elif name == "B":
        _section_agent_decisions()

    # ── Recent trades ────────────────────────────────────────────────
    trades = load_trades()
    port_trades = trades[trades["portfolio"] == name] if not trades.empty else pd.DataFrame()
    if not port_trades.empty:
        st.subheader("Recent Trades")
        st.dataframe(
            port_trades.head(20).style.format({
                "shares": "{:.0f}",
                "price": "${:.2f}",
                "total": "${:,.0f}",
            }),
            use_container_width=True,
            hide_index=True,
        )


def _section_momentum() -> None:
    """Momentum rankings section for Portfolio A."""
    rankings = load_momentum_rankings()
    if rankings.empty:
        return

    st.subheader("Momentum Rankings (63-day)")

    fig = px.bar(
        rankings.head(15),
        x="ticker",
        y="return_63d",
        color="return_63d",
        color_continuous_scale="RdYlGn",
        labels={"return_63d": "63d Return (%)", "ticker": "Ticker"},
    )
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Full Rankings Table"):
        st.dataframe(
            rankings.style.format({"return_63d": "{:.2f}%"}),
            use_container_width=True,
            hide_index=True,
        )


def _section_agent_decisions() -> None:
    """AI agent decisions section for Portfolio B."""
    decisions = load_agent_decisions()
    if decisions.empty:
        return

    st.subheader("Claude AI Decisions")

    for _, row in decisions.head(5).iterrows():
        with st.expander(f"{row['date']} — {row['model']} ({row['tokens']} tokens)"):
            st.markdown(f"**Reasoning:** {row['reasoning'][:500]}")
            try:
                trades = json.loads(row["proposed_trades"])
                if trades:
                    st.json(trades)
                else:
                    st.write("No trades proposed.")
            except (json.JSONDecodeError, TypeError):
                st.write(row["proposed_trades"])


# ── Trade log page ───────────────────────────────────────────────────────────


def page_trade_log() -> None:
    """Full trade history across all portfolios."""
    st.title("Trade Log")

    trades = load_trades()
    if trades.empty:
        no_data_warning()
        return

    # Filters
    col1, col2 = st.columns(2)
    with col1:
        portfolio_filter = st.multiselect(
            "Portfolio", ["A", "B"], default=["A", "B"]
        )
    with col2:
        side_filter = st.multiselect(
            "Side", ["BUY", "SELL"], default=["BUY", "SELL"]
        )

    filtered = trades[
        trades["portfolio"].isin(portfolio_filter)
        & trades["side"].isin(side_filter)
    ]

    # Summary metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Trades", len(filtered))
    buys = filtered[filtered["side"] == "BUY"]["total"].sum()
    sells = filtered[filtered["side"] == "SELL"]["total"].sum()
    c2.metric("Total Bought", f"${buys:,.0f}")
    c3.metric("Total Sold", f"${sells:,.0f}")
    c4.metric("Net Flow", f"${sells - buys:,.0f}")

    # Trade table
    st.dataframe(
        filtered.style.format({
            "shares": "{:.0f}",
            "price": "${:.2f}",
            "total": "${:,.0f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    # Volume by ticker
    if not filtered.empty:
        st.subheader("Trade Volume by Ticker")
        vol = filtered.groupby("ticker")["total"].sum().sort_values(ascending=False).head(15)
        fig = px.bar(
            x=vol.index,
            y=vol.values,
            labels={"x": "Ticker", "y": "Volume ($)"},
        )
        fig.update_layout(height=350, yaxis_tickprefix="$")
        st.plotly_chart(fig, use_container_width=True)


# ── Page router ──────────────────────────────────────────────────────────────

if page == "Overview":
    page_overview()
elif page == "Portfolio A":
    page_portfolio("A")
elif page == "Portfolio B":
    page_portfolio("B")
elif page == "Trade Log":
    page_trade_log()
