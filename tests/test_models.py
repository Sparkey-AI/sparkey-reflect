"""Tests for core data models."""

from datetime import datetime, timezone

from sparkey_reflect.core.models import (
    AnalysisResult,
    ConversationTurn,
    InsightCategory,
    InsightSeverity,
    ReflectInsight,
    ReflectReport,
    Session,
    SessionType,
    ToolType,
    TrendDirection,
)


class TestEnums:
    def test_tool_type_values(self):
        assert ToolType.CLAUDE_CODE.value == "claude_code"
        assert ToolType.CURSOR.value == "cursor"
        assert ToolType.COPILOT.value == "copilot"

    def test_session_type_values(self):
        assert SessionType.CODING.value == "coding"
        assert SessionType.DEBUGGING.value == "debugging"
        assert SessionType.UNKNOWN.value == "unknown"

    def test_insight_severity_ordering(self):
        severities = [s.value for s in InsightSeverity]
        assert "critical" in severities
        assert "info" in severities

    def test_trend_direction_values(self):
        assert TrendDirection.IMPROVING.value == "improving"
        assert TrendDirection.INSUFFICIENT_DATA.value == "insufficient_data"


class TestSession:
    def test_total_tokens(self, make_session):
        session = make_session()
        session.total_input_tokens = 500
        session.total_output_tokens = 1500
        assert session.total_tokens == 2000

    def test_turn_count(self, make_session, make_turn):
        session = make_session(turns=[
            make_turn(role="user", content="hello"),
            make_turn(role="assistant", content="hi"),
            make_turn(role="user", content="fix bug"),
        ])
        assert session.turn_count == 3

    def test_user_turn_count(self, make_session, make_turn):
        session = make_session(turns=[
            make_turn(role="user", content="hello"),
            make_turn(role="assistant", content="hi"),
            make_turn(role="user", content="fix bug"),
        ])
        assert session.user_turn_count == 2
        assert session.assistant_turn_count == 1

    def test_tool_use_count(self, make_session, make_turn):
        session = make_session(turns=[
            make_turn(role="assistant", content="running", tool_calls=[
                {"name": "Read"}, {"name": "Edit"},
            ]),
            make_turn(role="assistant", content="done", tool_calls=[
                {"name": "Bash"},
            ]),
        ])
        assert session.tool_use_count == 3

    def test_empty_session_properties(self):
        session = Session(session_id="empty", tool=ToolType.CLAUDE_CODE)
        assert session.total_tokens == 0
        assert session.turn_count == 0
        assert session.user_turn_count == 0
        assert session.tool_use_count == 0


class TestReflectInsight:
    def test_to_dict(self, make_insight):
        insight = make_insight(
            category=InsightCategory.TOOL_MASTERY,
            severity=InsightSeverity.WARNING,
        )
        d = insight.to_dict()
        assert d["category"] == "tool_mastery"
        assert d["severity"] == "warning"
        assert d["title"] == "Test Insight"
        assert d["recommendation"] == "Try this"

    def test_to_dict_with_created_at(self, make_insight):
        ts = datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc)
        insight = make_insight()
        insight.created_at = ts
        d = insight.to_dict()
        assert d["created_at"] == "2026-01-15T12:00:00+00:00"

    def test_to_dict_without_created_at(self, make_insight):
        d = make_insight().to_dict()
        assert d["created_at"] is None


class TestReflectReport:
    def test_to_dict(self, make_result, make_insight, now):
        report = ReflectReport(
            tool=ToolType.CLAUDE_CODE,
            period_start=now,
            period_end=now,
            overall_score=72.5,
            results=[make_result(score=80.0, metrics={"specificity": 18.5})],
            insights=[make_insight()],
            session_count=10,
            total_turns=50,
            total_tokens=5000,
            trends={"prompt_quality": TrendDirection.IMPROVING},
        )
        d = report.to_dict()
        assert d["overall_score"] == 72.5
        assert d["session_count"] == 10
        assert d["trends"]["prompt_quality"] == "improving"
        assert len(d["results"]) == 1
        assert d["results"][0]["score"] == 80.0
        assert d["results"][0]["metrics"]["specificity"] == 18.5
        assert len(d["insights"]) == 1

    def test_to_dict_empty(self, now):
        report = ReflectReport(
            tool=ToolType.CLAUDE_CODE,
            period_start=now,
            period_end=now,
            overall_score=0,
        )
        d = report.to_dict()
        assert d["results"] == []
        assert d["insights"] == []
        assert d["trends"] == {}
