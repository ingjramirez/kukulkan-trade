"""Tests for ContextManager.build_cached_system_prompt()."""

from src.agent.context_manager import ContextManager


def test_cached_prompt_returns_list():
    cm = ContextManager()
    result = cm.build_cached_system_prompt(pinned_context="## Test Context")
    assert isinstance(result, list)
    assert len(result) > 0


def test_identity_block_is_first():
    cm = ContextManager()
    result = cm.build_cached_system_prompt(pinned_context="test")
    first = result[0]
    assert first["type"] == "text"
    assert "Kukulkan Trade" in first["text"]


def test_identity_block_has_cache_control():
    cm = ContextManager()
    result = cm.build_cached_system_prompt(pinned_context="test")
    assert result[0]["cache_control"] == {"type": "ephemeral"}


def test_pinned_context_has_cache_control():
    cm = ContextManager()
    result = cm.build_cached_system_prompt(pinned_context="## My Context\nSome data")
    # Last block should be the pinned context with cache_control
    last = result[-1]
    assert "My Context" in last["text"]
    assert last["cache_control"] == {"type": "ephemeral"}


def test_strategy_directive_present():
    cm = ContextManager()
    result = cm.build_cached_system_prompt(
        pinned_context="context",
        strategy_directive="Be conservative and cautious.",
    )
    texts = [b["text"] for b in result]
    joined = " ".join(texts)
    assert "Be conservative and cautious" in joined


def test_strategy_directive_no_cache_control():
    cm = ContextManager()
    result = cm.build_cached_system_prompt(
        pinned_context="context",
        strategy_directive="Be aggressive.",
    )
    # Strategy block should be the middle block (index 1) without cache_control
    strategy_block = result[1]
    assert "Be aggressive" in strategy_block["text"]
    assert "cache_control" not in strategy_block


def test_no_pinned_context():
    cm = ContextManager()
    result = cm.build_cached_system_prompt(pinned_context="", strategy_directive="test")
    # Should have identity + strategy, no pinned context block
    assert len(result) == 2


def test_no_strategy_directive():
    cm = ContextManager()
    result = cm.build_cached_system_prompt(pinned_context="test context", strategy_directive="")
    # Should have identity + pinned context only
    assert len(result) == 2
