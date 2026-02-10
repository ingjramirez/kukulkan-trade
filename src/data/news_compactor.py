"""News compactor: 4-stage pipeline to produce dense, token-efficient news context.

Stage 1: RELEVANCE FILTER — keep only articles about held tickers, top movers, or macro events
Stage 2: CLUSTER BY EVENT — group articles about the same story into clusters
Stage 3: RANK BY ACTIONABILITY — score each cluster by trading relevance
Stage 4: FORMAT — produce dense pipe-delimited output (~250 tokens vs ~2000 raw)
"""

from __future__ import annotations

import re

import structlog

from src.data.news_article import NewsArticle, NewsCluster

log = structlog.get_logger()

# ── Signal classification keywords ──────────────────────────────────────────

_POS_KEYWORDS = frozenset({
    "beat", "beats", "surge", "surges", "upgrade", "upgraded", "raise",
    "raised", "rally", "rallies", "soar", "soars", "record", "strong",
    "jumps", "gains", "outperform", "bullish", "positive", "high",
    "all-time", "momentum", "growth", "optimism", "buy",
})

_NEG_KEYWORDS = frozenset({
    "miss", "misses", "crash", "crashes", "downgrade", "downgraded",
    "cut", "cuts", "layoff", "layoffs", "fall", "falls", "drop",
    "drops", "decline", "declines", "plunge", "plunges", "warns",
    "warning", "sell", "selloff", "bearish", "negative", "weak",
    "slump", "tumble", "fear", "fears", "loss", "loses",
})

_MACRO_KEYWORDS = frozenset({
    "fed", "federal reserve", "rate", "rates", "inflation", "gdp",
    "jobs", "employment", "unemployment", "tariff", "tariffs",
    "recession", "china", "trade war", "treasury", "yield", "cpi",
    "ppi", "fomc", "powell", "dollar", "economy", "economic",
})

_EVENT_KEYWORDS = frozenset({
    "merger", "acquisition", "acquires", "acquired", "ipo",
    "buyback", "spinoff", "spin-off", "split", "dividend",
    "bankruptcy", "restructuring", "deal", "takeover",
})

_EARNINGS_KEYWORDS = frozenset({
    "earnings", "revenue", "profit", "eps", "quarter", "q1", "q2",
    "q3", "q4", "guidance", "forecast", "outlook",
})

# Words to strip for headline compression
_NOISE_WORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "has", "have", "had",
    "that", "this", "with", "for", "its", "says", "said", "in", "of",
    "on", "to", "at", "by", "and", "or", "but", "be", "been", "will",
    "from", "as", "it", "he", "she", "they", "we", "our", "their",
    "his", "her", "about", "after", "before",
})


def _headline_words(headline: str) -> set[str]:
    """Extract significant words from a headline."""
    return {
        w for w in headline.lower().split()
        if w not in _NOISE_WORDS and len(w) > 1
    }


def _headlines_overlap(h1: str, h2: str, threshold: float = 0.50) -> bool:
    """Check if two headlines describe the same event."""
    words1 = _headline_words(h1)
    words2 = _headline_words(h2)
    if not words1 or not words2:
        return False
    smaller = min(len(words1), len(words2))
    if smaller == 0:
        return False
    overlap = len(words1 & words2)
    return (overlap / smaller) >= threshold


def classify_signal(headline: str, sentiment: float | None = None) -> str:
    """Classify a headline into POS, NEG, MACRO, EVENT, or INFO.

    Args:
        headline: Article headline text.
        sentiment: Optional sentiment score (-1.0 to 1.0).

    Returns:
        Signal string: POS, NEG, MACRO, EVENT, or INFO.
    """
    lower = headline.lower()
    words = set(re.findall(r"\b\w+\b", lower))

    # Check sentiment first if strong
    if sentiment is not None:
        if sentiment > 0.3:
            return "POS"
        if sentiment < -0.3:
            return "NEG"

    # Macro takes priority (regime-relevant)
    if words & _MACRO_KEYWORDS or any(kw in lower for kw in _MACRO_KEYWORDS if " " in kw):
        return "MACRO"

    # Corporate events
    if words & _EVENT_KEYWORDS:
        return "EVENT"

    # Positive signals (earnings-related boost)
    if words & _POS_KEYWORDS:
        return "POS"
    if words & _EARNINGS_KEYWORDS and words & _POS_KEYWORDS:
        return "POS"

    # Negative signals
    if words & _NEG_KEYWORDS:
        return "NEG"

    return "INFO"


def compress_headline(headline: str, max_words: int = 15) -> str:
    """Compress a headline by removing noise words and capping length.

    Args:
        headline: Raw headline text.
        max_words: Maximum words in output.

    Returns:
        Compressed headline string.
    """
    words = headline.split()
    compressed = [w for w in words if w.lower() not in _NOISE_WORDS]
    if not compressed:
        compressed = words  # fallback to original if all removed
    return " ".join(compressed[:max_words])


class NewsCompactor:
    """4-stage pipeline: Filter → Cluster → Rank → Format."""

    def __init__(self, max_clusters: int = 8) -> None:
        self._max_clusters = max_clusters
        self._last_universe: set[str] | None = None

    def compact(
        self,
        articles: list[NewsArticle],
        held_tickers: list[str] | None = None,
        top_movers: list[str] | None = None,
        universe_tickers: set[str] | None = None,
    ) -> str:
        """Run the full 4-stage compaction pipeline.

        Args:
            articles: Raw articles from NewsAggregator.
            held_tickers: Currently held portfolio tickers.
            top_movers: Tickers with largest recent price moves.
            universe_tickers: Full set of known universe tickers for discovery filtering.

        Returns:
            Dense pipe-delimited news context string.
        """
        held = held_tickers or []
        movers = top_movers or []
        self._last_universe = universe_tickers

        # Stage 1: Relevance filter
        relevant, discovery_articles = self._filter_relevant(
            articles, held, movers, universe_tickers,
        )

        # Stage 2: Cluster by event
        clusters = self._cluster_by_event(relevant)

        # Stage 3: Rank by actionability
        ranked = self._rank_clusters(clusters, held)

        # Stage 3.5: Cluster and rank discovery articles
        discovery_clusters: list[NewsCluster] = []
        if discovery_articles:
            discovery_clusters = self._cluster_by_event(discovery_articles)
            discovery_clusters = self._rank_discovery(discovery_clusters)

        # Stage 4: Format
        return self._format(ranked, discovery_clusters)

    def _filter_relevant(
        self,
        articles: list[NewsArticle],
        held_tickers: list[str],
        top_movers: list[str],
        universe_tickers: set[str] | None = None,
    ) -> tuple[list[NewsArticle], list[NewsArticle]]:
        """Stage 1: Keep articles about held/movers/macro; surface discovery articles.

        Args:
            articles: All fetched articles.
            held_tickers: Currently held tickers.
            top_movers: Biggest price movers today.
            universe_tickers: Full universe set. Articles with non-universe tickers
                go to discovery list.

        Returns:
            Tuple of (relevant articles, discovery articles).
        """
        priority_tickers = set(held_tickers + top_movers)
        universe = universe_tickers or set()
        result: list[NewsArticle] = []
        discovery: list[NewsArticle] = []

        for article in articles:
            if not article.headline:
                continue

            # Check ticker relevance
            ticker_match = any(t in priority_tickers for t in article.tickers)

            # Check macro relevance
            lower = article.headline.lower()
            macro_match = any(kw in lower for kw in _MACRO_KEYWORDS)

            if ticker_match or macro_match:
                result.append(article)
            elif universe and article.tickers:
                # Check if any ticker is outside the universe
                has_non_universe = any(t not in universe for t in article.tickers)
                if has_non_universe:
                    discovery.append(article)

        log.debug(
            "news_filter",
            input=len(articles),
            relevant=len(result),
            discovery=len(discovery),
        )
        return result, discovery

    def _cluster_by_event(
        self, articles: list[NewsArticle],
    ) -> list[NewsCluster]:
        """Stage 2: Group articles about the same event.

        Simple approach: shared tickers + >50% headline word overlap = same event.

        Args:
            articles: Filtered articles.

        Returns:
            List of NewsCluster objects.
        """
        clusters: list[NewsCluster] = []

        for article in articles:
            merged = False
            for cluster in clusters:
                rep = cluster.representative
                # Check headline overlap
                if _headlines_overlap(article.headline, rep.headline):
                    cluster.source_count += 1
                    # Merge tickers
                    merged_tickers = set(cluster.all_tickers + article.tickers)
                    cluster.all_tickers = list(merged_tickers)
                    # Keep the article with the longer summary as representative
                    if len(article.summary) > len(rep.summary):
                        cluster.representative = article
                    merged = True
                    break

            if not merged:
                clusters.append(NewsCluster(
                    representative=article,
                    source_count=1,
                    all_tickers=list(article.tickers),
                ))

        # Classify signals
        for cluster in clusters:
            cluster.signal = classify_signal(
                cluster.representative.headline,
                cluster.representative.sentiment,
            )

        log.debug("news_clustered", articles=len(articles), clusters=len(clusters))
        return clusters

    def _rank_clusters(
        self,
        clusters: list[NewsCluster],
        held_tickers: list[str],
    ) -> list[NewsCluster]:
        """Stage 3: Score each cluster by actionability.

        Scoring:
        - +30 if mentions a held ticker
        - +20 if earnings/M&A/downgrade/upgrade keyword
        - +15 if macro keyword
        - +min(sources, 5) * 5 for multi-source confirmation
        - +5 if sentiment score available

        Args:
            clusters: Clustered news events.
            held_tickers: Currently held tickers.

        Returns:
            Clusters sorted by score (descending), capped at max_clusters.
        """
        held_set = set(held_tickers)

        for cluster in clusters:
            score = 0
            headline_lower = cluster.representative.headline.lower()
            headline_words = set(re.findall(r"\b\w+\b", headline_lower))

            # Held ticker bonus
            if any(t in held_set for t in cluster.all_tickers):
                score += 30

            # Earnings/M&A keyword bonus
            if headline_words & (_EARNINGS_KEYWORDS | _EVENT_KEYWORDS):
                score += 20

            # Macro keyword bonus
            if headline_words & _MACRO_KEYWORDS or any(
                kw in headline_lower for kw in _MACRO_KEYWORDS if " " in kw
            ):
                score += 15

            # Multi-source confirmation
            score += min(cluster.source_count, 5) * 5

            # Sentiment available bonus
            if cluster.representative.sentiment is not None:
                score += 5

            cluster.score = score

        # Sort by score descending, cap at max_clusters
        ranked = sorted(clusters, key=lambda c: c.score, reverse=True)
        return ranked[:self._max_clusters]

    def _rank_discovery(
        self,
        clusters: list[NewsCluster],
        max_items: int = 3,
    ) -> list[NewsCluster]:
        """Rank discovery clusters by actionability and return top items.

        Scoring:
        - +20 if earnings or corporate event keywords
        - +5 per additional source (multi-source confirmation)
        - +5 if sentiment score available

        Args:
            clusters: Clustered discovery articles.
            max_items: Maximum discovery items to return.

        Returns:
            Top discovery clusters sorted by score.
        """
        for cluster in clusters:
            score = 0
            headline_lower = cluster.representative.headline.lower()
            headline_words = set(re.findall(r"\b\w+\b", headline_lower))

            # Earnings/event keyword bonus
            if headline_words & (_EARNINGS_KEYWORDS | _EVENT_KEYWORDS):
                score += 20

            # Multi-source confirmation
            score += min(cluster.source_count, 5) * 5

            # Sentiment available bonus
            if cluster.representative.sentiment is not None:
                score += 5

            cluster.score = score

        ranked = sorted(clusters, key=lambda c: c.score, reverse=True)
        return ranked[:max_items]

    def _pick_discovery_ticker(
        self, cluster: NewsCluster, universe_tickers: set[str],
    ) -> str:
        """Pick the best non-universe ticker from a discovery cluster.

        Args:
            cluster: Discovery news cluster.
            universe_tickers: Known universe tickers to exclude.

        Returns:
            A non-universe ticker string, or first ticker as fallback.
        """
        non_universe = [t for t in cluster.all_tickers if t and t not in universe_tickers]
        if non_universe:
            return non_universe[0]
        # Fallback to any ticker
        tickers = [t for t in cluster.all_tickers if t]
        return tickers[0] if tickers else "NEW"

    def _format(
        self,
        clusters: list[NewsCluster],
        discovery_clusters: list[NewsCluster] | None = None,
    ) -> str:
        """Stage 4: Format as dense pipe-delimited output.

        Target: ~250 tokens for the full news section.

        Format: TICKER|SIGNAL|EVENT|#SRC

        Args:
            clusters: Ranked and filtered clusters.
            discovery_clusters: Optional discovery clusters (non-universe tickers).

        Returns:
            Formatted string ready for agent prompt.
        """
        if not clusters and not discovery_clusters:
            return ""

        lines: list[str] = []

        if clusters:
            lines.append("TICKER|SIGNAL|EVENT|#SRC")
            for cluster in clusters:
                ticker = self._pick_display_ticker(cluster)
                signal = cluster.signal
                headline = compress_headline(cluster.representative.headline)
                sources = cluster.source_count
                lines.append(f"{ticker}|{signal}|{headline}|{sources}")

        if discovery_clusters:
            lines.append("")
            lines.append("== DISCOVERY (not in universe) ==")
            for cluster in discovery_clusters:
                ticker = self._pick_discovery_ticker(
                    cluster, self._last_universe or set(),
                )
                signal = cluster.signal
                headline = compress_headline(cluster.representative.headline)
                sources = cluster.source_count
                lines.append(f"{ticker}|{signal}|{headline}|{sources}")

        return "\n".join(lines)

    def _pick_display_ticker(self, cluster: NewsCluster) -> str:
        """Choose the best ticker to display for a cluster.

        Prefers specific tickers over ETFs, uses first article ticker as fallback.

        Args:
            cluster: The news cluster.

        Returns:
            Ticker string, or "MKT" for market-wide news.
        """
        tickers = [t for t in cluster.all_tickers if t]
        if not tickers:
            # Check for macro signal
            if cluster.signal == "MACRO":
                return "MACRO"
            return "MKT"

        # Prefer individual stocks over ETFs for display
        etfs = {"XLK", "XLF", "XLV", "XLE", "XLI", "XLY", "XLP", "XLU", "XLB",
                "XLRE", "QQQ", "SMH", "XBI", "IWM", "EFA", "EEM", "TLT", "HYG",
                "GDX", "ARKK", "SH", "PSQ", "TBF", "GLD", "SLV", "USO"}
        non_etf = [t for t in tickers if t not in etfs]
        if non_etf:
            return non_etf[0]
        return tickers[0]
