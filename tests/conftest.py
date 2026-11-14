import pytest


@pytest.fixture(autouse=True)
def _reset_abuse_limits():
    """Rate limits and quotas are module-level state; start every test from zero."""
    from app.core.security import abuse_guard, audio_rate_limit, chat_rate_limit

    for limiter in (chat_rate_limit, audio_rate_limit):
        limiter.reset()
    abuse_guard.reset()
    yield
