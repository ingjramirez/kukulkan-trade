"""Portfolio B: Sector Rotation + Macro + Contrarian strategy.

7-factor composite score with regime detection (BULL/ROTATION/NEUTRAL/BEAR).
Includes value tilt, crowding detection, BTC risk signal, and anti-tech-bubble rules.
"""

from datetime import date

import numpy as np
import pandas as pd
import structlog

from config.risk_rules import RISK_RULES
from config.strategies import PORTFOLIO_B
from config.universe import PORTFOLIO_B_UNIVERSE
from src.analysis.momentum import calculate_momentum
from src.analysis.technical import compute_rsi
from src.storage.models import (
    CompositeScoreRow,
    OrderSide,
    PortfolioName,
    Regime,
    TradeSchema,
)

log = structlog.get_logger()


def _normalize_scores(series: pd.Series) -> pd.Series:
    """Min-max normalize to [0, 1]. Handles constant series."""
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series(0.5, index=series.index)
    return (series - mn) / (mx - mn)


class RegimeDetector:
    """Detect the current market regime from macro + price signals."""

    def detect(
        self,
        spy_closes: pd.Series,
        yield_curve: float | None = None,
        vix: float | None = None,
    ) -> Regime:
        """Classify the current regime.

        Rules:
        - BEAR: SPY below 200-day SMA AND (inverted yield curve OR VIX > 30)
        - BULL: SPY above 50-day SMA AND VIX < 20
        - ROTATION: SPY above 200-day SMA but below 50-day SMA
        - NEUTRAL: everything else

        Args:
            spy_closes: S&P 500 close price series (>= 200 days preferred).
            yield_curve: 10Y-2Y spread (negative = inverted).
            vix: Current VIX value.

        Returns:
            Regime enum value.
        """
        if len(spy_closes) < 200:
            log.warning("insufficient_spy_data_for_regime", rows=len(spy_closes))
            return Regime.NEUTRAL

        current = spy_closes.iloc[-1]
        sma_50 = spy_closes.iloc[-50:].mean()
        sma_200 = spy_closes.iloc[-200:].mean()

        inverted_curve = yield_curve is not None and yield_curve < 0
        high_vix = vix is not None and vix > 30
        low_vix = vix is not None and vix < 20

        below_200 = current < sma_200
        above_200 = current >= sma_200
        above_50 = current >= sma_50
        below_50 = current < sma_50

        if below_200 and (inverted_curve or high_vix):
            regime = Regime.BEAR
        elif above_50 and low_vix:
            regime = Regime.BULL
        elif above_200 and below_50:
            regime = Regime.ROTATION
        else:
            regime = Regime.NEUTRAL

        log.info(
            "regime_detected",
            regime=regime.value,
            spy=round(current, 2),
            sma_50=round(sma_50, 2),
            sma_200=round(sma_200, 2),
            yield_curve=yield_curve,
            vix=vix,
        )
        return regime


class CompositeScorer:
    """Calculate the 7-factor composite score for Portfolio B."""

    def __init__(self, config: type(PORTFOLIO_B) = PORTFOLIO_B) -> None:
        self._cfg = config

    def score_momentum(self, closes: pd.DataFrame) -> pd.Series:
        """Factor 1: Momentum — 63-day returns, normalized.

        Args:
            closes: Price DataFrame for Portfolio B universe.

        Returns:
            Series of normalized momentum scores (0-1), indexed by ticker.
        """
        rankings = calculate_momentum(closes, lookback=63, skip=5)
        if rankings.empty:
            return pd.Series(dtype=float)
        scores = rankings.set_index("ticker")["return_63d"]
        return _normalize_scores(scores)

    def score_rsi_contrarian(self, closes: pd.DataFrame) -> pd.Series:
        """Factor 2: RSI Contrarian — favor oversold tickers.

        Inverted RSI: lower RSI = higher score (contrarian buy signal).

        Args:
            closes: Price DataFrame.

        Returns:
            Series of normalized contrarian scores (0-1), indexed by ticker.
        """
        rsi_values = {}
        for ticker in closes.columns:
            rsi = compute_rsi(closes[ticker])
            if not rsi.empty and not pd.isna(rsi.iloc[-1]):
                rsi_values[ticker] = rsi.iloc[-1]

        if not rsi_values:
            return pd.Series(dtype=float)

        rsi_series = pd.Series(rsi_values)
        # Invert: low RSI = high score (contrarian)
        return _normalize_scores(100 - rsi_series)

    def score_volume_breakout(self, volumes: pd.DataFrame) -> pd.Series:
        """Factor 4: Volume breakout — recent volume vs 20-day average.

        Args:
            volumes: Volume DataFrame with tickers as columns.

        Returns:
            Series of normalized volume breakout scores.
        """
        if len(volumes) < 20:
            return pd.Series(dtype=float)

        avg_20 = volumes.iloc[-20:].mean()
        recent = volumes.iloc[-5:].mean()
        ratio = recent / avg_20.replace(0, np.nan)
        return _normalize_scores(ratio.dropna())

    def score_value_tilt(self, closes: pd.DataFrame) -> pd.Series:
        """Factor 5: Value tilt — favor tickers below their 200-day SMA.

        Distance below SMA200 = higher value score (mean reversion).

        Args:
            closes: Price DataFrame (>= 200 days).

        Returns:
            Series of normalized value tilt scores.
        """
        if len(closes) < 200:
            return pd.Series(dtype=float)

        sma_200 = closes.iloc[-200:].mean()
        current = closes.iloc[-1]
        # Negative = below SMA200 = higher value tilt
        distance = (sma_200 - current) / sma_200
        return _normalize_scores(distance.dropna())

    def score_crowding(self, volumes: pd.DataFrame) -> pd.Series:
        """Factor 6: Anti-crowding — penalize excessively crowded names.

        Unusually high volume relative to history suggests crowding.
        We invert so low crowding = high score.

        Args:
            volumes: Volume DataFrame.

        Returns:
            Series of normalized anti-crowding scores.
        """
        if len(volumes) < 60:
            return pd.Series(dtype=float)

        avg_60 = volumes.iloc[-60:].mean()
        recent = volumes.iloc[-5:].mean()
        crowding = recent / avg_60.replace(0, np.nan)
        # Invert: low crowding = high score
        return _normalize_scores(1 / crowding.dropna().replace(0, np.nan)).dropna()

    def score_btc_risk(self, btc_closes: pd.Series) -> float:
        """Factor 7: BTC risk signal — uniform score based on BTC trend.

        Returns a single score applied to all tickers:
        - BTC down > 20% over 63 days → risk-off (low score)
        - BTC up → risk-on (high score)

        Args:
            btc_closes: Close price series for BTC proxy (IBIT).

        Returns:
            Score between 0 and 1.
        """
        if len(btc_closes) < 64:
            return 0.5  # neutral if insufficient data

        ret = (btc_closes.iloc[-1] - btc_closes.iloc[-63]) / btc_closes.iloc[-63]

        if ret < RISK_RULES.btc_crash_threshold_pct:
            return 0.0  # full risk-off
        elif ret > 0:
            return min(1.0, 0.5 + ret)  # risk-on, capped at 1.0
        else:
            return 0.5 + (ret / abs(RISK_RULES.btc_crash_threshold_pct)) * 0.5

    def compute_composite(
        self,
        closes: pd.DataFrame,
        volumes: pd.DataFrame,
        btc_closes: pd.Series | None = None,
        regime: Regime = Regime.NEUTRAL,
    ) -> pd.DataFrame:
        """Compute the full 7-factor composite score.

        Args:
            closes: Price DataFrame for Portfolio B universe.
            volumes: Volume DataFrame (same tickers).
            btc_closes: BTC proxy close prices.
            regime: Current market regime.

        Returns:
            DataFrame with columns: ticker, momentum_score, rsi_contrarian_score,
            macro_regime_score, volume_breakout_score, value_tilt_score,
            crowding_score, btc_risk_score, composite_score.
            Sorted by composite_score descending.
        """
        cfg = self._cfg
        tickers = [t for t in PORTFOLIO_B_UNIVERSE if t in closes.columns]
        if not tickers:
            return pd.DataFrame()

        # Calculate each factor
        f_momentum = self.score_momentum(closes[tickers])
        f_rsi = self.score_rsi_contrarian(closes[tickers])
        f_volume = self.score_volume_breakout(volumes[[t for t in tickers if t in volumes.columns]])
        f_value = self.score_value_tilt(closes[tickers])
        f_crowding = self.score_crowding(volumes[[t for t in tickers if t in volumes.columns]])

        # BTC risk is a single value applied uniformly
        btc_score = 0.5
        if btc_closes is not None and not btc_closes.empty:
            btc_score = self.score_btc_risk(btc_closes)

        # Regime score: boost defensive in BEAR, boost aggressive in BULL
        regime_scores = self._regime_adjustment(tickers, regime)

        # Combine all factors into a DataFrame
        result = pd.DataFrame(index=tickers)
        result["momentum_score"] = f_momentum.reindex(tickers, fill_value=0.5)
        result["rsi_contrarian_score"] = f_rsi.reindex(tickers, fill_value=0.5)
        result["macro_regime_score"] = regime_scores
        result["volume_breakout_score"] = f_volume.reindex(tickers, fill_value=0.5)
        result["value_tilt_score"] = f_value.reindex(tickers, fill_value=0.5)
        result["crowding_score"] = f_crowding.reindex(tickers, fill_value=0.5)
        result["btc_risk_score"] = btc_score

        # Weighted composite
        result["composite_score"] = (
            result["momentum_score"] * cfg.weight_momentum
            + result["rsi_contrarian_score"] * cfg.weight_rsi_contrarian
            + result["macro_regime_score"] * cfg.weight_macro_regime
            + result["volume_breakout_score"] * cfg.weight_volume_breakout
            + result["value_tilt_score"] * cfg.weight_value_tilt
            + result["crowding_score"] * cfg.weight_crowding
            + result["btc_risk_score"] * cfg.weight_btc_risk
        )

        result = result.reset_index().rename(columns={"index": "ticker"})
        result = result.sort_values("composite_score", ascending=False).reset_index(drop=True)

        log.info(
            "composite_scores_computed",
            regime=regime.value,
            top_ticker=result.iloc[0]["ticker"] if len(result) > 0 else None,
            top_score=round(result.iloc[0]["composite_score"], 4) if len(result) > 0 else None,
        )
        return result

    def _regime_adjustment(self, tickers: list[str], regime: Regime) -> pd.Series:
        """Generate regime-based score adjustments.

        BEAR: boost defensive assets, penalize tech.
        BULL: boost high-beta/growth.
        ROTATION: boost non-correlated assets.
        NEUTRAL: no adjustment (0.5 for all).
        """
        scores = pd.Series(0.5, index=tickers)

        if regime == Regime.BEAR:
            for t in tickers:
                if t in RISK_RULES.defensive_tickers:
                    scores[t] = 0.9
                elif t in RISK_RULES.tech_etfs:
                    scores[t] = 0.1
        elif regime == Regime.BULL:
            for t in tickers:
                if t in RISK_RULES.tech_etfs:
                    scores[t] = 0.8
                elif t in RISK_RULES.defensive_tickers:
                    scores[t] = 0.3
        elif regime == Regime.ROTATION:
            for t in tickers:
                if t in RISK_RULES.defensive_tickers:
                    scores[t] = 0.7
                elif t in RISK_RULES.tech_etfs:
                    scores[t] = 0.3

        return scores


class SectorRotationStrategy:
    """Full Portfolio B strategy: composite scoring + regime + risk rules."""

    def __init__(self) -> None:
        self._scorer = CompositeScorer()
        self._regime_detector = RegimeDetector()

    def analyze(
        self,
        closes: pd.DataFrame,
        volumes: pd.DataFrame,
        spy_closes: pd.Series | None = None,
        btc_closes: pd.Series | None = None,
        yield_curve: float | None = None,
        vix: float | None = None,
    ) -> tuple[pd.DataFrame, Regime]:
        """Run full analysis: detect regime, compute composite scores.

        Args:
            closes: Price DataFrame for the universe.
            volumes: Volume DataFrame.
            spy_closes: SPY close prices for regime detection.
            btc_closes: IBIT close prices for BTC risk signal.
            yield_curve: 10Y-2Y spread.
            vix: Current VIX value.

        Returns:
            Tuple of (scores DataFrame, detected Regime).
        """
        # Detect regime
        regime = Regime.NEUTRAL
        if spy_closes is not None and len(spy_closes) >= 200:
            regime = self._regime_detector.detect(spy_closes, yield_curve, vix)

        # Compute composite scores
        scores = self._scorer.compute_composite(
            closes=closes,
            volumes=volumes,
            btc_closes=btc_closes,
            regime=regime,
        )

        return scores, regime

    def select_positions(
        self,
        scores: pd.DataFrame,
        regime: Regime,
        top_n: int = PORTFOLIO_B.top_n,
    ) -> list[str]:
        """Select top N tickers, applying anti-tech-bubble rules.

        Args:
            scores: Composite scores DataFrame.
            regime: Current regime.
            top_n: Number of positions to hold.

        Returns:
            List of selected ticker symbols.
        """
        if scores.empty:
            return []

        selected: list[str] = []
        tech_count = 0
        max_tech = 1 if regime == Regime.BEAR else 2  # stricter in BEAR

        for _, row in scores.iterrows():
            if len(selected) >= top_n:
                break
            ticker = row["ticker"]
            is_tech = ticker in RISK_RULES.tech_etfs
            if is_tech and tech_count >= max_tech:
                continue
            selected.append(ticker)
            if is_tech:
                tech_count += 1

        # In BEAR, ensure at least one defensive holding
        if regime == Regime.BEAR:
            has_defensive = any(t in RISK_RULES.defensive_tickers for t in selected)
            if not has_defensive and len(scores) > 0:
                for _, row in scores.iterrows():
                    if row["ticker"] in RISK_RULES.defensive_tickers:
                        if len(selected) >= top_n:
                            selected[-1] = row["ticker"]
                        else:
                            selected.append(row["ticker"])
                        break

        log.info("positions_selected", tickers=selected, regime=regime.value)
        return selected

    def generate_trades(
        self,
        selected_tickers: list[str],
        current_positions: dict[str, float],
        cash: float,
        latest_prices: pd.Series,
    ) -> list[TradeSchema]:
        """Generate trade signals to move to target positions.

        Equal-weight allocation across selected tickers.

        Args:
            selected_tickers: Target tickers from select_positions().
            current_positions: Current ticker -> shares held.
            cash: Available cash.
            latest_prices: Most recent prices for all tickers.

        Returns:
            List of TradeSchema (sells first, then buys).
        """
        trades: list[TradeSchema] = []

        # Sell positions not in target
        for ticker, shares in current_positions.items():
            if ticker not in selected_tickers and shares > 0:
                price = latest_prices.get(ticker)
                if price is not None and not pd.isna(price):
                    trades.append(TradeSchema(
                        portfolio=PortfolioName.B,
                        ticker=ticker,
                        side=OrderSide.SELL,
                        shares=shares,
                        price=float(price),
                        reason="dropped from composite ranking",
                    ))

        # Calculate total available after sells
        sell_proceeds = sum(t.total for t in trades)
        total_available = cash + sell_proceeds

        # Add value of positions we're keeping
        for ticker in selected_tickers:
            if ticker in current_positions:
                price = latest_prices.get(ticker, 0)
                total_available += current_positions[ticker] * price

        # Equal-weight target allocation
        if not selected_tickers:
            return trades

        target_per_position = total_available / len(selected_tickers)

        for ticker in selected_tickers:
            price = latest_prices.get(ticker)
            if price is None or pd.isna(price) or price <= 0:
                continue

            current_shares = current_positions.get(ticker, 0)
            current_value = current_shares * price
            target_shares = int(target_per_position / price)
            delta = target_shares - current_shares

            if delta > 0:
                trades.append(TradeSchema(
                    portfolio=PortfolioName.B,
                    ticker=ticker,
                    side=OrderSide.BUY,
                    shares=float(delta),
                    price=float(price),
                    reason="composite ranking rebalance",
                ))
            elif delta < 0:
                trades.append(TradeSchema(
                    portfolio=PortfolioName.B,
                    ticker=ticker,
                    side=OrderSide.SELL,
                    shares=float(abs(delta)),
                    price=float(price),
                    reason="rebalance trim",
                ))

        return trades

    def scores_to_db_rows(
        self, scores: pd.DataFrame, score_date: date, regime: Regime
    ) -> list[CompositeScoreRow]:
        """Convert scores DataFrame to DB rows.

        Args:
            scores: Composite scores from compute_composite().
            score_date: Date for these scores.
            regime: Detected regime.

        Returns:
            List of CompositeScoreRow objects.
        """
        return [
            CompositeScoreRow(
                date=score_date,
                ticker=row["ticker"],
                momentum_score=row["momentum_score"],
                rsi_contrarian_score=row["rsi_contrarian_score"],
                macro_regime_score=row["macro_regime_score"],
                volume_breakout_score=row["volume_breakout_score"],
                value_tilt_score=row["value_tilt_score"],
                crowding_score=row["crowding_score"],
                btc_risk_score=row["btc_risk_score"],
                composite_score=row["composite_score"],
                regime=regime.value,
            )
            for _, row in scores.iterrows()
        ]
