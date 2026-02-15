"""Tests for session profiles mapping and budget override."""

from src.agent.session_profiles import SessionProfile, get_session_profile


def test_morning_maps_to_full():
    assert get_session_profile("morning") == SessionProfile.FULL


def test_midday_maps_to_light():
    assert get_session_profile("midday") == SessionProfile.LIGHT


def test_close_maps_to_light():
    assert get_session_profile("close") == SessionProfile.LIGHT


def test_event_maps_to_crisis():
    assert get_session_profile("event") == SessionProfile.CRISIS


def test_weekly_review_maps_to_review():
    assert get_session_profile("weekly_review") == SessionProfile.REVIEW


def test_unknown_trigger_defaults_to_light():
    assert get_session_profile("unknown_trigger") == SessionProfile.LIGHT


def test_budget_exhausted_overrides_full():
    assert get_session_profile("morning", budget_exhausted=True) == SessionProfile.BUDGET_SAVING


def test_budget_exhausted_overrides_crisis():
    assert get_session_profile("event", budget_exhausted=True) == SessionProfile.BUDGET_SAVING


def test_budget_not_exhausted_uses_normal_mapping():
    assert get_session_profile("morning", budget_exhausted=False) == SessionProfile.FULL


def test_session_profile_values():
    assert SessionProfile.FULL.value == "full"
    assert SessionProfile.LIGHT.value == "light"
    assert SessionProfile.CRISIS.value == "crisis"
    assert SessionProfile.REVIEW.value == "review"
    assert SessionProfile.BUDGET_SAVING.value == "budget_saving"
