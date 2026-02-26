"""
Insight Generator

Combines analyzer results with LLM-generated insights to produce actionable
recommendations. Numeric scoring stays local; insights come from Claude.
"""

import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

from sparkey_reflect.core.models import (
    AnalysisResult,
    InsightCategory,
    InsightSeverity,
    ReflectInsight,
    ReflectReport,
    RuleFileInfo,
    Session,
    TrendDirection,
    ToolType,
)
from sparkey_reflect.core.storage import ReflectStorage
from sparkey_reflect.insights.llm_generator import LLMInsightGenerator
from sparkey_reflect.insights.templates import (
    DAILY_DIGEST_TEMPLATE,
    REPORT_FOOTER,
    WEEKLY_DIGEST_TEMPLATE,
    format_insight_card,
)

logger = logging.getLogger(__name__)

# Weights for overall score computation (sum ~1.0)
# completion_patterns only applies to Copilot; when present, weights are
# renormalized at runtime so the sum stays 1.0.
ANALYZER_WEIGHTS = {
    "prompt_quality": 0.20,
    "conversation_flow": 0.20,
    "session_patterns": 0.15,
    "context_management": 0.15,
    "tool_usage": 0.10,
    "rule_file": 0.05,
    "outcome_tracker": 0.15,
    "completion_patterns": 0.15,
}


class InsightGenerator:
    """Generates coaching insights from analysis results."""

    def __init__(
        self,
        storage: Optional[ReflectStorage] = None,
        use_llm: bool = True,
    ):
        self.storage = storage
        self.use_llm = use_llm
        self._llm_generator = LLMInsightGenerator() if use_llm else None

    def generate_report(
        self,
        results: List[AnalysisResult],
        tool: ToolType,
        period_start: datetime,
        period_end: datetime,
        sessions_meta: Optional[Dict] = None,
        sessions: Optional[List[Session]] = None,
        rule_files: Optional[List[RuleFileInfo]] = None,
    ) -> ReflectReport:
        """
        Generate a complete report from analyzer results.

        Args:
            results: List of AnalysisResult from each analyzer.
            tool: Which AI tool this report covers.
            period_start: Report period start.
            period_end: Report period end.
            sessions_meta: Optional dict with session_count, total_turns, etc.
            sessions: Conversation sessions for LLM analysis.
            rule_files: Rule files for LLM analysis.

        Returns:
            ReflectReport with overall score, insights, and trends.
        """
        # Compute weighted overall score
        overall_score = self._compute_overall_score(results)

        # Compute trends if storage is available
        trends = {}
        if self.storage:
            for r in results:
                trend = self._compute_trend(r.analyzer_key, r.score, tool.value)
                trends[r.analyzer_key] = trend

        # Generate insights via LLM (or return empty if disabled)
        all_insights = []
        overall_assessment = None
        if self.use_llm and self._llm_generator and sessions:
            llm_data = self._llm_generator.generate_insights(
                results=results,
                sessions=sessions,
                rule_files=rule_files,
                trends=trends,
            )
            all_insights = self._llm_generator.parse_insights(llm_data)
            overall_assessment = llm_data.get("overall_assessment")

        # Enrich insights with trend info
        for insight in all_insights:
            if insight.metric_key and insight.metric_key in trends:
                insight.trend = trends[insight.metric_key]
            insight.created_at = datetime.now(timezone.utc)

        # Sort insights: critical first, then warnings, suggestions, info
        severity_order = {
            InsightSeverity.CRITICAL: 0,
            InsightSeverity.WARNING: 1,
            InsightSeverity.SUGGESTION: 2,
            InsightSeverity.INFO: 3,
        }
        all_insights.sort(key=lambda i: severity_order.get(i.severity, 99))

        meta = sessions_meta or {}
        report = ReflectReport(
            tool=tool,
            period_start=period_start,
            period_end=period_end,
            overall_score=overall_score,
            results=results,
            insights=all_insights,
            session_count=meta.get("session_count", sum(r.session_count for r in results[:1])),
            total_turns=meta.get("total_turns", 0),
            total_tokens=meta.get("total_tokens", 0),
            total_duration_minutes=meta.get("total_duration_minutes", 0),
            trends=trends,
            overall_assessment=overall_assessment,
            created_at=datetime.now(timezone.utc),
        )

        return report

    def format_weekly_digest(self, report: ReflectReport) -> str:
        """Format a report as a readable weekly digest with full insights."""
        scores_lines = []
        for r in sorted(report.results, key=lambda x: x.score, reverse=True):
            bar = self._score_bar(r.score)
            scores_lines.append(f"  {r.analyzer_name:<25} {bar} {r.score:.0f}/100")

        insights_lines = []
        for i, insight in enumerate(report.insights[:8], 1):
            insights_lines.append(format_insight_card(insight, i))

        trends_lines = []
        for key, direction in report.trends.items():
            arrow = {
                "improving": "^", "declining": "v", "stable": "=", "insufficient_data": "?"
            }.get(direction.value, "?")
            trends_lines.append(f"  {arrow} {key}: {direction.value}")

        assessment_section = ""
        if report.overall_assessment:
            assessment_section = f"\n{report.overall_assessment}\n"

        body = WEEKLY_DIGEST_TEMPLATE.format(
            period_start=report.period_start.strftime("%Y-%m-%d"),
            period_end=report.period_end.strftime("%Y-%m-%d"),
            tool=report.tool.value,
            overall_score=f"{report.overall_score:.0f}",
            overall_assessment_section=assessment_section,
            session_count=report.session_count,
            total_duration=report.total_duration_minutes,
            total_tokens=report.total_tokens,
            scores_section="\n".join(scores_lines) if scores_lines else "  No data",
            insights_section="\n".join(insights_lines) if insights_lines else "  No insights",
            trends_section="\n".join(trends_lines) if trends_lines else "  Not enough data yet",
        )
        return body + REPORT_FOOTER

    def format_daily_digest(self, report: ReflectReport) -> str:
        """Format a report as a daily digest with full insights."""
        insights_lines = []
        for i, insight in enumerate(report.insights[:5], 1):
            insights_lines.append(format_insight_card(insight, i))

        assessment_section = ""
        if report.overall_assessment:
            assessment_section = f"\n{report.overall_assessment}\n"

        body = DAILY_DIGEST_TEMPLATE.format(
            date=report.period_end.strftime("%Y-%m-%d"),
            tool=report.tool.value,
            overall_score=f"{report.overall_score:.0f}",
            overall_assessment_section=assessment_section,
            session_count=report.session_count,
            total_duration=report.total_duration_minutes,
            insights_section="\n".join(insights_lines) if insights_lines else "  All good!",
        )
        return body + REPORT_FOOTER

    # =========================================================================
    # Internal Helpers
    # =========================================================================

    def _compute_overall_score(self, results: List[AnalysisResult]) -> float:
        """Compute weighted overall score."""
        weighted_sum = 0.0
        weight_sum = 0.0
        for r in results:
            weight = ANALYZER_WEIGHTS.get(r.analyzer_key, 0.1)
            weighted_sum += r.score * weight
            weight_sum += weight
        return weighted_sum / weight_sum if weight_sum > 0 else 0

    def _compute_trend(self, analyzer_key: str, current_score: float,
                       tool: str) -> TrendDirection:
        """Determine trend direction by comparing to historical scores."""
        if not self.storage:
            return TrendDirection.INSUFFICIENT_DATA

        history = self.storage.get_score_history(analyzer_key, tool, days=30)
        if len(history) < 3:
            return TrendDirection.INSUFFICIENT_DATA

        # Compare current to average of older entries
        older_scores = [h["score"] for h in history[:-1]]
        avg_older = sum(older_scores) / len(older_scores)
        diff = current_score - avg_older

        if diff > 5:
            return TrendDirection.IMPROVING
        elif diff < -5:
            return TrendDirection.DECLINING
        return TrendDirection.STABLE

    def _score_bar(self, score: float, width: int = 20) -> str:
        """Generate a text-based score bar."""
        filled = int(score / 100 * width)
        return "[" + "#" * filled + "." * (width - filled) + "]"
