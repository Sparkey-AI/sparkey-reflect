"""Tests for message templates and formatting."""

from sparkey_reflect.core.models import InsightCategory, InsightSeverity, ReflectInsight
from sparkey_reflect.insights.templates import (
    REPORT_FOOTER,
    SEVERITY_ICONS,
    format_insight_card,
)


class TestSeverityIcons:
    def test_all_severities_have_icons(self):
        assert "critical" in SEVERITY_ICONS
        assert "warning" in SEVERITY_ICONS
        assert "suggestion" in SEVERITY_ICONS
        assert "info" in SEVERITY_ICONS


class TestFormatInsightCard:
    def test_basic_card(self):
        insight = ReflectInsight(
            category=InsightCategory.PROMPT_ENGINEERING,
            title="Use more specific prompts",
            severity=InsightSeverity.SUGGESTION,
            recommendation="Include file paths and line numbers",
            evidence="5 of 10 prompts were vague",
        )
        card = format_insight_card(insight, 1)
        assert "SUGGESTION" in card
        assert "Use more specific prompts" in card
        assert "Include file paths" in card
        assert "5 of 10" in card

    def test_critical_severity(self):
        insight = ReflectInsight(
            category=InsightCategory.SESSION_HABITS,
            title="Session fatigue detected",
            severity=InsightSeverity.CRITICAL,
            recommendation="Take breaks",
            evidence="Late-night sessions",
        )
        card = format_insight_card(insight, 1)
        assert "CRITICAL" in card
        assert "!!" in card  # critical icon

    def test_no_evidence(self):
        insight = ReflectInsight(
            category=InsightCategory.TOOL_MASTERY,
            title="Try MCP tools",
            severity=InsightSeverity.INFO,
            recommendation="Explore MCP",
            evidence="",
        )
        card = format_insight_card(insight, 1)
        assert "INFO" in card
        # No evidence line when evidence is empty
        assert "Evidence:" not in card


class TestReportFooter:
    def test_footer_contains_url(self):
        assert "sparkey.ai" in REPORT_FOOTER

    def test_footer_contains_teams_edition(self):
        assert "Teams edition" in REPORT_FOOTER
