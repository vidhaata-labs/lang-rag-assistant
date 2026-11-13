from datetime import datetime, timedelta

from lang_rag_assistant.ui.components import format_recency


def test_recency_labels():
    now = datetime(2026, 11, 13, 12, 0)
    assert format_recency(now - timedelta(hours=2), now) == "Updated 2 hours ago"
    assert format_recency(now - timedelta(days=1), now) == "Updated yesterday"
    assert format_recency(now - timedelta(days=3), now) == "Updated 3 days ago"


def test_recency_singular_hour():
    now = datetime(2026, 11, 13, 12, 0)
    assert format_recency(now - timedelta(hours=1), now) == "Updated 1 hour ago"
