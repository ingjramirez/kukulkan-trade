"""Local ticker signal engine — batch scoring and ranking.

Runs every 10 min during market hours. Uses ONLY cached data from SQLite —
NO external API calls. Pure pandas/numpy/ta computation.

This is an ATTENTION ROUTING system: it tells the agent "look here" by
surfacing rank velocity (movers), technical alerts, and composite scores.
The agent still makes all trade decisions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd
import structlog
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands

from src.storage.models import TickerSignalRow

log = structlog.get_logger()


# ── Signal weights (must sum to 1.0) ─────────────────────────────────────────

SIGNAL_WEIGHTS: dict[str, float] = {
    "momentum_20d": 0.20,
    "momentum_63d": 0.15,
    "rsi_position": 0.15,
    "macd_histogram": 0.10,
    "sma_trend": 0.15,
    "bollinger_pct_b": 0.10,
    "volume_surge": 0.15,
}

# Alert thresholds
ALERT_RANK_JUMP = 10
ALERT_RSI_OVERSOLD = 30
ALERT_RSI_OVERBOUGHT = 70
ALERT_VOLUME_SPIKE = 2.0
ALERT_BB_UPPER = 1.0
ALERT_BB_LOWER = 0.0


@dataclass
class TickerSignal:
    """Signal snapshot for a single ticker."""

    ticker: str
    composite_score: float
    rank: int
    prev_rank: int | None
    rank_velocity: float
    momentum_20d: float
    momentum_63d: float
    rsi: float
    macd_histogram: float
    sma_trend_score: float
    bollinger_pct_b: float
    volume_ratio: float
    alerts: list[str] = field(default_factory=list)
    scored_at: datetime = field(default_factory=lambda: datetime.utcnow())


class SignalEngine:
    """Batch ticker ranking engine — pure pandas/numpy/ta computation.

    Scores all tickers on 7 quantitative signals, z-score normalizes across
    the universe, and produces a composite rank (0-100). Detects rank velocity
    and technical alerts (golden cross, RSI oversold, volume spike, etc.).
    """

    def __init__(self) -> None:
        self._prev_ranks: dict[str, dict[str, int]] = {}  # tenant_id -> {ticker: rank}
        self._prev_run_time: dict[str, datetime] = {}
        self._prev_indicators: dict[str, dict[str, dict]] = {}  # tenant_id -> {ticker: {rsi, sma20, sma50}}

    async def run(
        self,
        tenant_id: str,
        closes: pd.DataFrame,
        volumes: pd.DataFrame,
    ) -> list[TickerSignal]:
        """Score and rank all tickers. Detect changes from previous run.

        Args:
            tenant_id: Tenant for state isolation.
            closes: DataFrame (dates × tickers) of close prices.
            volumes: DataFrame (dates × tickers) of volumes.

        Returns:
            List of TickerSignal sorted by rank (best first).
        """
        if closes.empty or len(closes) < 30:
            return []

        tickers = [t for t in closes.columns if closes[t].dropna().shape[0] >= 15]
        if not tickers:
            return []

        # Compute individual signal scores
        signals = {
            "momentum_20d": self._compute_momentum_20d(closes, tickers),
            "momentum_63d": self._compute_momentum_63d(closes, tickers),
            "rsi_position": self._compute_rsi_score(closes, tickers),
            "macd_histogram": self._compute_macd_score(closes, tickers),
            "sma_trend": self._compute_sma_trend(closes, tickers),
            "bollinger_pct_b": self._compute_bollinger_score(closes, tickers),
            "volume_surge": self._compute_volume_score(closes, volumes, tickers),
        }

        # Z-score normalize each signal across the universe
        z_signals: dict[str, pd.Series] = {}
        for name, series in signals.items():
            std = series.std()
            if std > 0:
                z_signals[name] = (series - series.mean()) / std
            else:
                z_signals[name] = series * 0

        # Weighted composite
        composite = pd.Series(0.0, index=tickers)
        for name, weight in SIGNAL_WEIGHTS.items():
            if name in z_signals:
                composite += z_signals[name].reindex(tickers, fill_value=0) * weight

        # Normalize to 0-100
        c_min, c_max = composite.min(), composite.max()
        if c_max > c_min:
            composite = ((composite - c_min) / (c_max - c_min)) * 100
        else:
            composite = pd.Series(50.0, index=tickers)

        # Rank (1 = highest score)
        ranks = composite.rank(ascending=False, method="min").astype(int)

        # Rank velocity
        prev_ranks = self._prev_ranks.get(tenant_id, {})
        prev_time = self._prev_run_time.get(tenant_id)
        now = datetime.utcnow()
        hours_elapsed = 1.0
        if prev_time:
            elapsed = (now - prev_time).total_seconds() / 3600
            hours_elapsed = max(elapsed, 0.1)

        # Raw indicators for alert detection
        raw_indicators: dict[str, dict] = {}
        for ticker in tickers:
            raw_indicators[ticker] = {
                "rsi": _raw_rsi(closes, ticker),
                "sma20": _raw_sma(closes, ticker, 20),
                "sma50": _raw_sma(closes, ticker, 50),
                "bollinger_pct_b": _raw_bollinger_pctb(closes, ticker),
                "volume_ratio": _raw_volume_ratio(closes, volumes, ticker),
            }

        prev_ind = self._prev_indicators.get(tenant_id, {})

        # Build results
        results: list[TickerSignal] = []
        for ticker in tickers:
            rank = int(ranks.get(ticker, len(tickers)))
            prev_rank = prev_ranks.get(ticker)
            velocity = ((prev_rank - rank) / hours_elapsed) if prev_rank is not None else 0.0

            alerts = _detect_alerts(
                rank,
                prev_rank,
                hours_elapsed,
                raw_indicators.get(ticker, {}),
                prev_ind.get(ticker, {}),
            )

            results.append(
                TickerSignal(
                    ticker=ticker,
                    composite_score=round(float(composite.get(ticker, 50)), 1),
                    rank=rank,
                    prev_rank=prev_rank,
                    rank_velocity=round(velocity, 1),
                    momentum_20d=round(float(signals["momentum_20d"].get(ticker, 0)), 4),
                    momentum_63d=round(float(signals["momentum_63d"].get(ticker, 0)), 4),
                    rsi=round(raw_indicators.get(ticker, {}).get("rsi", 50.0), 1),
                    macd_histogram=round(float(signals["macd_histogram"].get(ticker, 0)), 4),
                    sma_trend_score=round(float(signals["sma_trend"].get(ticker, 0)), 1),
                    bollinger_pct_b=round(raw_indicators.get(ticker, {}).get("bollinger_pct_b", 0.5), 3),
                    volume_ratio=round(raw_indicators.get(ticker, {}).get("volume_ratio", 1.0), 2),
                    alerts=alerts,
                    scored_at=now,
                )
            )

        results.sort(key=lambda s: s.rank)

        # Update state for next run
        self._prev_ranks[tenant_id] = {s.ticker: s.rank for s in results}
        self._prev_run_time[tenant_id] = now
        self._prev_indicators[tenant_id] = raw_indicators

        log.info(
            "signal_engine_run",
            tenant_id=tenant_id,
            tickers=len(results),
            alerts=sum(1 for s in results if s.alerts),
            top_ticker=results[0].ticker if results else None,
        )

        return results

    # ── Signal computation (each returns pd.Series indexed by ticker) ─────

    @staticmethod
    def _compute_momentum_20d(closes: pd.DataFrame, tickers: list[str]) -> pd.Series:
        """20-day return."""
        if len(closes) < 21:
            return pd.Series(0.0, index=tickers)
        cur = closes[tickers].iloc[-1]
        prev = closes[tickers].iloc[-21]
        ret = (cur - prev) / prev
        return ret.fillna(0)

    @staticmethod
    def _compute_momentum_63d(closes: pd.DataFrame, tickers: list[str]) -> pd.Series:
        """63-day return, skip last 5 days (momentum reversal avoidance)."""
        if len(closes) < 69:
            return pd.Series(0.0, index=tickers)
        end = closes[tickers].iloc[-6]
        start = closes[tickers].iloc[-69]
        ret = (end - start) / start
        return ret.fillna(0)

    @staticmethod
    def _compute_rsi_score(closes: pd.DataFrame, tickers: list[str]) -> pd.Series:
        """RSI distance from 50 — extremes score higher (both oversold and overbought).

        Intentionally non-directional: used for attention routing, not buy/sell signals.
        RSI 30 and RSI 70 score equally because both deserve investigation.
        """
        scores: dict[str, float] = {}
        for t in tickers:
            rsi_val = _raw_rsi(closes, t)
            scores[t] = abs(rsi_val - 50) / 50  # 0-1
        return pd.Series(scores)

    @staticmethod
    def _compute_macd_score(closes: pd.DataFrame, tickers: list[str]) -> pd.Series:
        """MACD histogram — positive = bullish."""
        scores: dict[str, float] = {}
        for t in tickers:
            try:
                series = closes[t].dropna()
                if len(series) < 35:
                    scores[t] = 0.0
                    continue
                macd = MACD(series)
                hist = macd.macd_diff()
                val = hist.iloc[-1] if not hist.empty and pd.notna(hist.iloc[-1]) else 0.0
                scores[t] = float(val)
            except Exception:
                scores[t] = 0.0
        return pd.Series(scores)

    @staticmethod
    def _compute_sma_trend(closes: pd.DataFrame, tickers: list[str]) -> pd.Series:
        """Count of SMAs (20/50/200) that price is above. Range: 0-3."""
        scores: dict[str, float] = {}
        for t in tickers:
            series = closes[t].dropna()
            if len(series) < 20:
                scores[t] = 0.0
                continue
            price = float(series.iloc[-1])
            count = 0
            for period in [20, 50, 200]:
                if len(series) >= period:
                    sma = float(series.rolling(period).mean().iloc[-1])
                    if price > sma:
                        count += 1
            scores[t] = float(count)
        return pd.Series(scores)

    @staticmethod
    def _compute_bollinger_score(closes: pd.DataFrame, tickers: list[str]) -> pd.Series:
        """Bollinger %B — position within bands."""
        scores: dict[str, float] = {}
        for t in tickers:
            scores[t] = _raw_bollinger_pctb(closes, t)
        return pd.Series(scores)

    @staticmethod
    def _compute_volume_score(
        closes: pd.DataFrame,
        volumes: pd.DataFrame,
        tickers: list[str],
    ) -> pd.Series:
        """Today's volume vs 20-day average."""
        scores: dict[str, float] = {}
        for t in tickers:
            scores[t] = _raw_volume_ratio(closes, volumes, t)
        return pd.Series(scores)


# ── Raw indicator helpers (module-level for reuse) ────────────────────────────


def _raw_rsi(closes: pd.DataFrame, ticker: str) -> float:
    """Compute current RSI for a single ticker."""
    try:
        series = closes[ticker].dropna()
        if len(series) < 15:
            return 50.0
        rsi = RSIIndicator(series, window=14).rsi()
        val = rsi.iloc[-1]
        return float(val) if pd.notna(val) else 50.0
    except Exception:
        return 50.0


def _raw_sma(closes: pd.DataFrame, ticker: str, period: int) -> float:
    """Compute SMA for a single ticker."""
    try:
        series = closes[ticker].dropna()
        if len(series) < period:
            return float(series.iloc[-1]) if len(series) > 0 else 0.0
        return float(series.rolling(period).mean().iloc[-1])
    except Exception:
        return 0.0


def _raw_bollinger_pctb(closes: pd.DataFrame, ticker: str) -> float:
    """Compute Bollinger %B for a single ticker."""
    try:
        series = closes[ticker].dropna()
        if len(series) < 20:
            return 0.5
        bb = BollingerBands(series, window=20)
        pctb = bb.bollinger_pband().iloc[-1]
        return float(pctb) if pd.notna(pctb) else 0.5
    except Exception:
        return 0.5


def _raw_volume_ratio(closes: pd.DataFrame, volumes: pd.DataFrame, ticker: str) -> float:
    """Compute volume ratio (today / 20d avg) for a single ticker."""
    try:
        if ticker not in volumes.columns:
            return 1.0
        vol = volumes[ticker].dropna()
        if len(vol) < 21:
            return 1.0
        avg_20 = vol.iloc[-21:-1].mean()
        if avg_20 == 0:
            return 1.0
        return float(vol.iloc[-1] / avg_20)
    except Exception:
        return 1.0


def _detect_alerts(
    rank: int,
    prev_rank: int | None,
    hours_elapsed: float,
    indicators: dict,
    prev_indicators: dict,
) -> list[str]:
    """Detect alert conditions for a ticker."""
    alerts: list[str] = []

    # Rank jump
    if prev_rank is not None and abs(prev_rank - rank) >= ALERT_RANK_JUMP:
        direction = "up" if rank < prev_rank else "down"
        alerts.append(f"rank_jump_{direction}_{abs(prev_rank - rank)}")

    # RSI crosses (only on transition, not static)
    rsi = indicators.get("rsi", 50.0)
    prev_rsi = prev_indicators.get("rsi", 50.0)
    if rsi <= ALERT_RSI_OVERSOLD and prev_rsi > ALERT_RSI_OVERSOLD:
        alerts.append("rsi_oversold_cross")
    if rsi >= ALERT_RSI_OVERBOUGHT and prev_rsi < ALERT_RSI_OVERBOUGHT:
        alerts.append("rsi_overbought_cross")

    # Golden cross / death cross (SMA20 vs SMA50)
    sma20 = indicators.get("sma20", 0.0)
    sma50 = indicators.get("sma50", 0.0)
    prev_sma20 = prev_indicators.get("sma20", 0.0)
    prev_sma50 = prev_indicators.get("sma50", 0.0)
    if sma20 > sma50 and prev_sma20 <= prev_sma50 and prev_sma20 > 0:
        alerts.append("golden_cross")
    if sma20 < sma50 and prev_sma20 >= prev_sma50 and prev_sma20 > 0:
        alerts.append("death_cross")

    # Volume spike
    vol_ratio = indicators.get("volume_ratio", 1.0)
    if vol_ratio >= ALERT_VOLUME_SPIKE:
        alerts.append(f"volume_spike_{vol_ratio:.1f}x")

    # Bollinger breakout
    pctb = indicators.get("bollinger_pct_b", 0.5)
    if pctb > ALERT_BB_UPPER:
        alerts.append("bollinger_upper_breakout")
    if pctb < ALERT_BB_LOWER:
        alerts.append("bollinger_lower_breakout")

    return alerts


def format_signals_for_agent(
    signals: list[TickerSignal],
    held_tickers: set[str],
    top_n: int = 10,
) -> str:
    """Format signal data for injection into agent trigger messages.

    Replaces build_universe_opportunities() output with live signal data.
    """
    if not signals:
        return ""

    lines = ["## Universe Signal Rankings (updated every 10 min)"]

    # Biggest movers (rank velocity)
    movers = sorted(
        [s for s in signals if s.rank_velocity != 0],
        key=lambda s: abs(s.rank_velocity),
        reverse=True,
    )[:5]
    if movers:
        lines.append("Biggest movers (rank velocity):")
        for s in movers:
            direction = "+" if s.rank_velocity > 0 else ""
            prev = f" (was {s.prev_rank})" if s.prev_rank is not None else ""
            alert_str = f" — {', '.join(s.alerts)}" if s.alerts else ""
            held = " [HELD]" if s.ticker in held_tickers else ""
            lines.append(
                f"  {s.ticker}: rank {s.rank}{prev} vel={direction}{s.rank_velocity:.0f}/hr "
                f"RSI {s.rsi:.0f}, vol {s.volume_ratio:.1f}x{alert_str}{held}"
            )

    # Top non-held by composite score
    non_held = [s for s in signals if s.ticker not in held_tickers][:top_n]
    if non_held:
        lines.append("Top non-held by composite score:")
        for s in non_held[:5]:
            alert_str = f" | {', '.join(s.alerts)}" if s.alerts else ""
            lines.append(
                f"  {s.rank}. {s.ticker} (score {s.composite_score:.0f}) "
                f"mom20d {s.momentum_20d:+.1%}, RSI {s.rsi:.0f}, "
                f"SMA {s.sma_trend_score:.0f}/3{alert_str}"
            )

    # All alerts
    all_alerts = [s for s in signals if s.alerts]
    if all_alerts:
        lines.append("Alerts triggered:")
        for s in all_alerts:
            held = " [HELD]" if s.ticker in held_tickers else ""
            lines.append(f"  {s.ticker}: {', '.join(s.alerts)}{held}")

    lines.append("Use get_signal_rankings tool for full data. Investigate movers with get_batch_technicals.")
    return "\n".join(lines)


def db_rows_to_signals(rows: list[TickerSignalRow]) -> list[TickerSignal]:
    """Convert ORM rows back to TickerSignal dataclasses (for formatting)."""
    return [
        TickerSignal(
            ticker=r.ticker,
            composite_score=r.composite_score,
            rank=r.rank,
            prev_rank=r.prev_rank,
            rank_velocity=r.rank_velocity,
            momentum_20d=r.momentum_20d or 0.0,
            momentum_63d=r.momentum_63d or 0.0,
            rsi=r.rsi or 50.0,
            macd_histogram=r.macd_histogram or 0.0,
            sma_trend_score=r.sma_trend_score or 0.0,
            bollinger_pct_b=r.bollinger_pct_b or 0.5,
            volume_ratio=r.volume_ratio or 1.0,
            alerts=json.loads(r.alerts) if r.alerts else [],
            scored_at=r.scored_at if r.scored_at else datetime.utcnow(),
        )
        for r in rows
    ]


def signals_to_db_rows(tenant_id: str, signals: list[TickerSignal]) -> list[TickerSignalRow]:
    """Convert TickerSignal list to ORM rows for DB persistence."""
    return [
        TickerSignalRow(
            tenant_id=tenant_id,
            ticker=s.ticker,
            composite_score=s.composite_score,
            rank=s.rank,
            prev_rank=s.prev_rank,
            rank_velocity=s.rank_velocity,
            momentum_20d=s.momentum_20d,
            momentum_63d=s.momentum_63d,
            rsi=s.rsi,
            macd_histogram=s.macd_histogram,
            sma_trend_score=s.sma_trend_score,
            bollinger_pct_b=s.bollinger_pct_b,
            volume_ratio=s.volume_ratio,
            alerts=json.dumps(s.alerts),
            scored_at=s.scored_at,
        )
        for s in signals
    ]
