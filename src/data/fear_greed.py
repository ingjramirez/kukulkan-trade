"""Fear & Greed Index fetcher from alternative.me API.

Free, no API key required. Returns 0-100 value with classification.
Fetched twice daily (9:30 AM + 4:30 PM ET) and persisted to SentimentIndicatorRow.
"""

from __future__ import annotations

import json

import httpx
import structlog

log = structlog.get_logger()

FEAR_GREED_API_URL = "https://api.alternative.me/fng/"
INDICATOR_NAME = "fear_greed_index"


def _classify(value: int) -> str:
    """Classify Fear & Greed value into human-readable label."""
    if value <= 20:
        return "Extreme Fear"
    if value <= 40:
        return "Fear"
    if value <= 60:
        return "Neutral"
    if value <= 80:
        return "Greed"
    return "Extreme Greed"


def fetch_fear_greed() -> dict | None:
    """Fetch current Fear & Greed Index from alternative.me.

    Returns:
        Dict with value, classification, timestamp — or None on failure.
    """
    try:
        resp = httpx.get(FEAR_GREED_API_URL, params={"limit": "1"}, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        entries = data.get("data", [])
        if not entries:
            log.warning("fear_greed_empty_response")
            return None

        entry = entries[0]
        value = int(entry.get("value", 0))
        classification = entry.get("value_classification", _classify(value))
        timestamp = entry.get("timestamp", "")

        return {
            "value": value,
            "classification": classification,
            "timestamp": timestamp,
        }
    except (httpx.HTTPError, ValueError, KeyError) as e:
        log.warning("fear_greed_fetch_failed", error=str(e))
        return None


async def fetch_and_save(db: object, tenant_id: str) -> dict | None:
    """Fetch Fear & Greed and persist to database.

    Args:
        db: Database instance with save_sentiment_indicator method.
        tenant_id: Tenant UUID.

    Returns:
        The fetched data dict, or None on failure.
    """
    data = fetch_fear_greed()
    if data is None:
        return None

    try:
        await db.save_sentiment_indicator(
            tenant_id=tenant_id,
            name=INDICATOR_NAME,
            value=float(data["value"]),
            classification=data["classification"],
            sub_indicators=json.dumps({"timestamp": data["timestamp"]}),
        )
        log.info(
            "fear_greed_saved",
            tenant_id=tenant_id,
            value=data["value"],
            classification=data["classification"],
        )
    except Exception as e:
        log.warning("fear_greed_save_failed", tenant_id=tenant_id, error=str(e))

    return data


def format_for_context(value: float, classification: str) -> str:
    """Format Fear & Greed for injection into agent context.

    Args:
        value: Numeric F&G value (0-100).
        classification: Text classification.

    Returns:
        One-line string for trigger messages.
    """
    return f"Fear & Greed Index: {value:.0f}/100 ({classification})"
