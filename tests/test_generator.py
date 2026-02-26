"""Tests for the Insight Generator."""

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from sparkey_reflect.core.models import (
    AnalysisResult,
    ToolType,
    TrendDirection,
)
from sparkey_reflect.insights.generator import InsightGenerator, ANALYZER_WEIGHTS


@pytest.fixture
def generator():
    """Generator with LLM disabled."""
    return InsightGenerator(storage=None, use_llm=False)


class TestAnalyzerWeights:
    def test_weights_sum_correctly(self):
        """Non-copilot weights (excluding completion_patterns) should sum to ~1.0."""
        non_copilot = {k: v for k, v in ANALYZER_WEIGHTS.items() if k != "completion_patterns"}
        assert sum(non_copilot.values()) == pytest.approx(1.0, abs=0.01)

    def test_outcome_tracker_weight_increased(self):
        assert ANALYZER_WEIGHTS["outcome_tracker"] == 0.20

    def test_session_patterns_weight_decreased(self):
        assert ANALYZER_WEIGHTS["session_patterns"] == 0.10

    def test_prompt_quality_unchanged(self):
        assert ANALYZER_WEIGHTS["prompt_quality"] == 0.20

    def test_conversation_flow_unchanged(self):
        assert ANALYZER_WEIGHTS["conversation_flow"] == 0.20


class TestComputeOverallScore:
    def test_single_result(self, generator, make_result):
        results = [make_result(key="prompt_quality", score=80.0)]
        score = generator._compute_overall_score(results)
        assert score == 80.0

    def test_weighted_average(self, generator, make_result):
        results = [
            make_result(key="prompt_quality", score=100.0),  # weight 0.20
            make_result(key="conversation_flow", score=0.0),  # weight 0.20
        ]
        score = generator._compute_overall_score(results)
        # (100 * 0.20 + 0 * 0.20) / 0.40 = 50.0
        assert score == pytest.approx(50.0)

    def test_empty_results(self, generator):
        assert generator._compute_overall_score([]) == 0

    def test_unknown_analyzer_gets_default_weight(self, generator, make_result):
        results = [make_result(key="unknown_analyzer", score=60.0)]
        score = generator._compute_overall_score(results)
        assert score == 60.0

    def test_outcome_tracker_has_higher_impact(self, generator, make_result):
        """Outcome tracker weight (0.20) now exceeds session_patterns (0.10)."""
        results_outcome_high = [
            make_result(key="outcome_tracker", score=100.0),
            make_result(key="session_patterns", score=0.0),
        ]
        results_outcome_low = [
            make_result(key="outcome_tracker", score=0.0),
            make_result(key="session_patterns", score=100.0),
        ]
        score_high = generator._compute_overall_score(results_outcome_high)
        score_low = generator._compute_overall_score(results_outcome_low)
        # outcome_tracker at 100 should produce higher overall than session_patterns at 100
        assert score_high > score_low


class TestComputeTrend:
    def test_insufficient_data_without_storage(self, generator):
        trend = generator._compute_trend("prompt_quality", 70.0, "claude_code")
        assert trend == TrendDirection.INSUFFICIENT_DATA

    def test_insufficient_data_with_short_history(self):
        storage = MagicMock()
        storage.get_score_history.return_value = [{"score": 50.0}]
        gen = InsightGenerator(storage=storage, use_llm=False)
        trend = gen._compute_trend("prompt_quality", 70.0, "claude_code")
        assert trend == TrendDirection.INSUFFICIENT_DATA

    def test_improving_trend(self):
        storage = MagicMock()
        storage.get_score_history.return_value = [
            {"score": 50.0}, {"score": 52.0}, {"score": 55.0}, {"score": 60.0},
        ]
        gen = InsightGenerator(storage=storage, use_llm=False)
        trend = gen._compute_trend("prompt_quality", 70.0, "claude_code")
        assert trend == TrendDirection.IMPROVING

    def test_declining_trend(self):
        storage = MagicMock()
        storage.get_score_history.return_value = [
            {"score": 80.0}, {"score": 78.0}, {"score": 75.0}, {"score": 70.0},
        ]
        gen = InsightGenerator(storage=storage, use_llm=False)
        trend = gen._compute_trend("prompt_quality", 60.0, "claude_code")
        assert trend == TrendDirection.DECLINING

    def test_stable_trend(self):
        storage = MagicMock()
        storage.get_score_history.return_value = [
            {"score": 70.0}, {"score": 71.0}, {"score": 69.0}, {"score": 70.0},
        ]
        gen = InsightGenerator(storage=storage, use_llm=False)
        trend = gen._compute_trend("prompt_quality", 71.0, "claude_code")
        assert trend == TrendDirection.STABLE


class TestScoreBar:
    def test_full_score(self, generator):
        bar = generator._score_bar(100)
        assert bar == "[" + "#" * 20 + "]"

    def test_zero_score(self, generator):
        bar = generator._score_bar(0)
        assert bar == "[" + "." * 20 + "]"

    def test_half_score(self, generator):
        bar = generator._score_bar(50)
        assert bar.count("#") == 10
        assert bar.count(".") == 10


class TestGenerateReport:
    def test_basic_report(self, generator, make_result, now):
        results = [
            make_result(key="prompt_quality", score=70.0),
            make_result(key="conversation_flow", score=65.0),
        ]
        report = generator.generate_report(
            results=results,
            tool=ToolType.CLAUDE_CODE,
            period_start=now,
            period_end=now,
        )
        assert report.overall_score > 0
        assert report.tool == ToolType.CLAUDE_CODE
        assert len(report.results) == 2

    def test_report_without_llm_has_no_insights(self, generator, make_result, now):
        results = [make_result()]
        report = generator.generate_report(
            results=results,
            tool=ToolType.CLAUDE_CODE,
            period_start=now,
            period_end=now,
        )
        assert len(report.insights) == 0
        assert report.overall_assessment is None


class TestFormatDigest:
    def test_weekly_digest_contains_sections(self, generator, make_result, now):
        from sparkey_reflect.core.models import ReflectReport
        report = ReflectReport(
            tool=ToolType.CLAUDE_CODE,
            period_start=now,
            period_end=now,
            overall_score=72.0,
            results=[make_result(key="prompt_quality", score=72.0)],
            session_count=10,
            total_duration_minutes=120.0,
            total_tokens=5000,
        )
        text = generator.format_weekly_digest(report)
        assert "Weekly Digest" in text
        assert "72/100" in text
        assert "sparkey.ai" in text

    def test_daily_digest_contains_sections(self, generator, make_result, now):
        from sparkey_reflect.core.models import ReflectReport
        report = ReflectReport(
            tool=ToolType.CLAUDE_CODE,
            period_start=now,
            period_end=now,
            overall_score=65.0,
            session_count=3,
            total_duration_minutes=45.0,
        )
        text = generator.format_daily_digest(report)
        assert "Daily Digest" in text
        assert "65/100" in text
