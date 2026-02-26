"""Tests for SQLite storage layer using in-memory database."""

from datetime import datetime, timezone, timedelta

import pytest

from sparkey_reflect.core.models import (
    AnalysisResult,
    InsightCategory,
    InsightSeverity,
    ReflectInsight,
    ReflectReport,
    Session,
    ToolType,
)
from sparkey_reflect.core.storage import ReflectStorage


@pytest.fixture
def storage(tmp_path):
    """Create a storage instance with a temp path."""
    from pathlib import Path
    db_path = tmp_path / "test.db"
    s = ReflectStorage(db_path=Path(db_path))
    yield s
    s.close()


class TestStorageSchema:
    def test_creates_tables(self, storage):
        cursor = storage.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        tables = {row[0] for row in cursor.fetchall()}
        assert "analysis_results" in tables
        assert "insights" in tables
        assert "reports" in tables
        assert "config" in tables


class TestConfig:
    def test_set_and_get(self, storage):
        storage.set_config("test_key", "test_value")
        assert storage.get_config("test_key") == "test_value"

    def test_get_missing_returns_default(self, storage):
        assert storage.get_config("missing") is None
        assert storage.get_config("missing", "fallback") == "fallback"

    def test_overwrite_config(self, storage):
        storage.set_config("key", "v1")
        storage.set_config("key", "v2")
        assert storage.get_config("key") == "v2"


class TestAnalysisResults:
    def test_save_and_retrieve(self, storage):
        result = AnalysisResult(
            analyzer_key="prompt_quality",
            analyzer_name="Prompt Quality",
            score=72.5,
            metrics={"specificity": 18.0},
            session_count=5,
        )
        row_id = storage.save_analysis_result(result, "claude_code")
        assert row_id > 0

        scores = storage.get_latest_scores("claude_code", limit=10)
        assert len(scores) >= 1
        assert scores[0]["analyzer_key"] == "prompt_quality"
        assert scores[0]["score"] == 72.5

    def test_score_history(self, storage):
        for i in range(5):
            result = AnalysisResult(
                analyzer_key="prompt_quality",
                analyzer_name="Prompt Quality",
                score=50.0 + i * 5,
                session_count=3,
            )
            storage.save_analysis_result(result, "claude_code")

        history = storage.get_score_history("prompt_quality", "claude_code", days=30)
        assert len(history) == 5


class TestReports:
    def test_save_and_get_latest(self, storage):
        now = datetime.now(timezone.utc)
        report = ReflectReport(
            tool=ToolType.CLAUDE_CODE,
            period_start=now - timedelta(days=7),
            period_end=now,
            overall_score=68.0,
            session_count=10,
        )
        row_id = storage.save_report(report)
        assert row_id > 0

        latest = storage.get_latest_report("claude_code")
        assert latest is not None
        assert latest["overall_score"] == 68.0


class TestTrends:
    def test_save_and_get_trend(self, storage):
        now = datetime.now(timezone.utc)
        for i in range(5):
            storage.save_trend_point(
                metric_key="prompt_quality",
                value=60.0 + i,
                tool="claude_code",
                measured_at=now - timedelta(days=5 - i),
            )

        trend = storage.get_trend("prompt_quality", "claude_code", days=30)
        assert len(trend) == 5

    def test_empty_trend(self, storage):
        trend = storage.get_trend("nonexistent", "claude_code", days=30)
        assert trend == []
