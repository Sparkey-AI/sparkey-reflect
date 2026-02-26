"""Shared test fixtures for Sparkey Reflect."""

from datetime import datetime, timezone, timedelta

import pytest

from sparkey_reflect.core.models import (
    AnalysisResult,
    ConversationTurn,
    InsightCategory,
    InsightSeverity,
    ReflectInsight,
    ReflectReport,
    RuleFileInfo,
    Session,
    SessionType,
    ToolType,
    TrendDirection,
)


@pytest.fixture
def now():
    return datetime.now(timezone.utc)


@pytest.fixture
def make_turn():
    """Factory for ConversationTurn objects."""

    def _make(
        role="user",
        content="Fix the bug in auth.py",
        file_references=None,
        has_error_context=False,
        has_code_snippet=False,
        tool_calls=None,
        tool_name=None,
        timestamp=None,
    ):
        return ConversationTurn(
            role=role,
            content=content,
            timestamp=timestamp,
            tool_calls=tool_calls or [],
            tool_name=tool_name,
            file_references=file_references or [],
            has_error_context=has_error_context,
            has_code_snippet=has_code_snippet,
        )

    return _make


@pytest.fixture
def make_session(make_turn, now):
    """Factory for Session objects with sensible defaults."""

    def _make(
        turns=None,
        tool=ToolType.CLAUDE_CODE,
        duration_minutes=30.0,
        session_type=SessionType.CODING,
        workspace_path="/home/dev/project",
        start_time=None,
        end_time=None,
        session_id="test-session-001",
    ):
        start = start_time or now - timedelta(hours=1)
        end = end_time or now
        return Session(
            session_id=session_id,
            tool=tool,
            turns=turns or [make_turn()],
            start_time=start,
            end_time=end,
            duration_minutes=duration_minutes,
            workspace_path=workspace_path,
            session_type=session_type,
            total_input_tokens=1000,
            total_output_tokens=2000,
        )

    return _make


@pytest.fixture
def make_result():
    """Factory for AnalysisResult objects."""

    def _make(key="prompt_quality", name="Prompt Quality", score=65.0, metrics=None):
        return AnalysisResult(
            analyzer_key=key,
            analyzer_name=name,
            score=score,
            metrics=metrics or {},
            session_count=5,
        )

    return _make


@pytest.fixture
def make_insight():
    """Factory for ReflectInsight objects."""

    def _make(
        category=InsightCategory.PROMPT_ENGINEERING,
        title="Test Insight",
        severity=InsightSeverity.SUGGESTION,
        recommendation="Try this",
        evidence="Based on data",
    ):
        return ReflectInsight(
            category=category,
            title=title,
            severity=severity,
            recommendation=recommendation,
            evidence=evidence,
        )

    return _make


@pytest.fixture
def make_rule_file():
    """Factory for RuleFileInfo objects."""

    def _make(
        exists=True,
        word_count=500,
        section_count=5,
        has_examples=True,
        has_constraints=True,
        has_project_context=True,
        has_style_guide=False,
        file_type="claude_md",
        raw_content="# CLAUDE.md\n\n## Rules\n- Use TypeScript\n- Follow conventions",
    ):
        return RuleFileInfo(
            file_path="/project/CLAUDE.md",
            file_type=file_type,
            tool=ToolType.CLAUDE_CODE,
            exists=exists,
            word_count=word_count,
            section_count=section_count,
            has_examples=has_examples,
            has_constraints=has_constraints,
            has_project_context=has_project_context,
            has_style_guide=has_style_guide,
            raw_content=raw_content,
        )

    return _make


@pytest.fixture
def sample_sessions(make_session, make_turn, now):
    """A realistic set of sessions for analyzer testing."""
    return [
        make_session(
            session_id="session-1",
            turns=[
                make_turn(
                    content="Fix the TypeError in api/auth.py line 42 where get_user returns None instead of raising NotFoundError",
                    file_references=["api/auth.py"],
                    has_error_context=True,
                ),
                make_turn(role="assistant", content="I'll fix the error handling in auth.py."),
                make_turn(content="Thanks, looks good"),
            ],
            start_time=now - timedelta(hours=2),
            end_time=now - timedelta(hours=1),
        ),
        make_session(
            session_id="session-2",
            turns=[
                make_turn(
                    content="Add a new endpoint POST /api/teams that creates a team with name and description fields. Use SQLAlchemy for the model.",
                    file_references=["api/teams.py", "core/schema/team.py"],
                    has_code_snippet=True,
                ),
                make_turn(role="assistant", content="I'll create the team endpoint and model."),
                make_turn(content="Also add validation for team name length"),
                make_turn(role="assistant", content="Added validation."),
                make_turn(content="Perfect, ship it"),
            ],
            start_time=now - timedelta(hours=4),
            end_time=now - timedelta(hours=3),
            duration_minutes=45.0,
        ),
        make_session(
            session_id="session-3",
            session_type=SessionType.DEBUGGING,
            turns=[
                make_turn(
                    content="help me debug this",
                ),
                make_turn(role="assistant", content="Could you share more details?"),
                make_turn(content="no wait, I meant the auth error"),
                make_turn(role="assistant", content="Let me look at auth."),
                make_turn(content="As I said before, the error is in the login function"),
                make_turn(role="assistant", content="Found it."),
            ],
            start_time=now - timedelta(hours=6),
            end_time=now - timedelta(hours=5),
            duration_minutes=20.0,
        ),
    ]
