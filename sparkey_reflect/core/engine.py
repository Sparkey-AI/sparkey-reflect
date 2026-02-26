"""
Reflect Engine

Main orchestrator that ties readers, analyzers, storage, and insight generation together.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

from sparkey_reflect.analyzers.base_analyzer import BaseReflectAnalyzer
from sparkey_reflect.analyzers.completion_patterns import CompletionPatternsAnalyzer
from sparkey_reflect.analyzers.context_management import ContextManagementAnalyzer
from sparkey_reflect.analyzers.conversation_flow import ConversationFlowAnalyzer
from sparkey_reflect.analyzers.outcome_tracker import OutcomeTrackerAnalyzer
from sparkey_reflect.analyzers.prompt_quality import PromptQualityAnalyzer
from sparkey_reflect.analyzers.registry import ReflectAnalyzerConfig, ReflectAnalyzerPresets
from sparkey_reflect.analyzers.rule_file import RuleFileAnalyzer
from sparkey_reflect.analyzers.session_patterns import SessionPatternsAnalyzer
from sparkey_reflect.analyzers.tool_usage import ToolUsageAnalyzer
from sparkey_reflect.core.models import (
    AnalysisResult,
    ReflectReport,
    RuleFileInfo,
    Session,
    ToolType,
)
from sparkey_reflect.core.storage import ReflectStorage
from sparkey_reflect.insights.generator import InsightGenerator
from sparkey_reflect.readers.base_reader import BaseReader
from sparkey_reflect.readers.claude_code_reader import ClaudeCodeReader
from sparkey_reflect.readers.copilot_reader import CopilotReader
from sparkey_reflect.readers.cursor_reader import CursorReader

logger = logging.getLogger(__name__)

# Mapping of analyzer keys to their implementing classes
ANALYZER_CLASSES: Dict[str, type] = {
    "prompt_quality": PromptQualityAnalyzer,
    "conversation_flow": ConversationFlowAnalyzer,
    "session_patterns": SessionPatternsAnalyzer,
    "context_management": ContextManagementAnalyzer,
    "tool_usage": ToolUsageAnalyzer,
    "rule_file": RuleFileAnalyzer,
    "outcome_tracker": OutcomeTrackerAnalyzer,
    "completion_patterns": CompletionPatternsAnalyzer,
}

# Mapping of tool types to reader classes
READER_CLASSES: Dict[ToolType, type] = {
    ToolType.CLAUDE_CODE: ClaudeCodeReader,
    ToolType.CURSOR: CursorReader,
    ToolType.COPILOT: CopilotReader,
}


class ReflectEngine:
    """Main orchestrator for the Reflect analysis pipeline."""

    def __init__(
        self,
        storage: Optional[ReflectStorage] = None,
        analyzer_config: Optional[ReflectAnalyzerConfig] = None,
        use_llm: bool = True,
    ):
        self.storage = storage or ReflectStorage()
        self.analyzer_config = analyzer_config
        self.insight_generator = InsightGenerator(storage=self.storage, use_llm=use_llm)
        self._readers: Dict[ToolType, BaseReader] = {}
        self._analyzers: Dict[str, BaseReflectAnalyzer] = {}

    # =========================================================================
    # Public API
    # =========================================================================

    def analyze(
        self,
        tool: Optional[ToolType] = None,
        days: int = 7,
        preset: Optional[str] = None,
        workspace_path: Optional[str] = None,
    ) -> ReflectReport:
        """
        Run analysis pipeline: read sessions -> analyze -> generate report.

        Args:
            tool: Which AI tool to analyze. None = auto-detect.
            days: Number of days of history to analyze.
            preset: Analyzer preset name (quick, coaching, full).
            workspace_path: Filter to a specific workspace.

        Returns:
            ReflectReport with scores, insights, and trends.
        """
        # Resolve tool
        if tool is None:
            tool = self._auto_detect_tool()
        logger.info("Analyzing %s for last %d days", tool.value, days)

        # Set up analyzer config
        config = self._resolve_config(preset, tool)

        # Read sessions
        since = datetime.now(timezone.utc) - timedelta(days=days)
        until = datetime.now(timezone.utc)
        reader = self._get_reader(tool)
        sessions = reader.read_sessions(since=since, until=until, workspace_path=workspace_path)
        logger.info("Read %d sessions from %s", len(sessions), tool.value)

        if not sessions:
            return ReflectReport(
                tool=tool,
                period_start=since,
                period_end=until,
                overall_score=0,
                created_at=datetime.now(timezone.utc),
            )

        # Read rule files
        rule_files = reader.read_rule_files(workspace_path=workspace_path)

        # Run analyzers
        results = self._run_analyzers(sessions, rule_files, config)

        # Compute session metadata
        sessions_meta = {
            "session_count": len(sessions),
            "total_turns": sum(s.turn_count for s in sessions),
            "total_tokens": sum(s.total_tokens for s in sessions),
            "total_duration_minutes": sum(s.duration_minutes for s in sessions),
        }

        # Generate report
        report = self.insight_generator.generate_report(
            results=results,
            tool=tool,
            period_start=since,
            period_end=until,
            sessions_meta=sessions_meta,
            sessions=sessions,
            rule_files=rule_files,
        )

        # Persist results
        self._persist(report, tool)

        return report

    def analyze_rules(
        self,
        tool: Optional[ToolType] = None,
        workspace_path: Optional[str] = None,
    ) -> List[RuleFileInfo]:
        """Analyze rule files for a tool without session analysis."""
        if tool is None:
            tool = self._auto_detect_tool()
        reader = self._get_reader(tool)
        return reader.read_rule_files(workspace_path=workspace_path)

    def get_status(self) -> List[Dict]:
        """Get availability status for all supported tools."""
        statuses = []
        for tool_type, reader_cls in READER_CLASSES.items():
            reader = reader_cls()
            statuses.append(reader.get_status())
        return statuses

    def get_trends(
        self,
        tool: Optional[ToolType] = None,
        metric_key: Optional[str] = None,
        days: int = 30,
    ) -> Dict[str, List[Dict]]:
        """Get trend data from storage."""
        tool_str = tool.value if tool else self._auto_detect_tool().value
        if metric_key:
            return {metric_key: self.storage.get_trend(metric_key, tool_str, days)}

        # Return all known analyzer trends
        trends = {}
        for key in ANALYZER_CLASSES:
            trend_data = self.storage.get_trend(key, tool_str, days)
            if trend_data:
                trends[key] = trend_data
        return trends

    def get_latest_report(self, tool: Optional[ToolType] = None) -> Optional[Dict]:
        """Get the most recent saved report."""
        tool_str = tool.value if tool else self._auto_detect_tool().value
        return self.storage.get_latest_report(tool_str)

    # =========================================================================
    # Internals
    # =========================================================================

    def _auto_detect_tool(self) -> ToolType:
        """Detect which AI tools are available and pick the best one."""
        for tool_type, reader_cls in READER_CLASSES.items():
            reader = reader_cls()
            if reader.is_available():
                return tool_type
        raise RuntimeError(
            "No AI coding tool data found. Supported tools: "
            + ", ".join(t.value for t in READER_CLASSES)
        )

    def _get_reader(self, tool: ToolType) -> BaseReader:
        if tool not in self._readers:
            reader_cls = READER_CLASSES.get(tool)
            if not reader_cls:
                raise ValueError(f"No reader available for {tool.value}")
            self._readers[tool] = reader_cls()
        return self._readers[tool]

    def _get_analyzer(self, key: str) -> BaseReflectAnalyzer:
        if key not in self._analyzers:
            analyzer_cls = ANALYZER_CLASSES.get(key)
            if not analyzer_cls:
                raise ValueError(f"Unknown analyzer: {key}")
            self._analyzers[key] = analyzer_cls()
        return self._analyzers[key]

    def _resolve_config(
        self, preset: Optional[str], tool: ToolType
    ) -> ReflectAnalyzerConfig:
        """Resolve which analyzers to run."""
        if self.analyzer_config:
            return self.analyzer_config

        preset_map = {
            "quick": ReflectAnalyzerPresets.quick,
            "coaching": ReflectAnalyzerPresets.coaching,
            "full": ReflectAnalyzerPresets.full,
            "copilot": ReflectAnalyzerPresets.copilot,
        }
        factory = preset_map.get(preset or "coaching", ReflectAnalyzerPresets.coaching)
        return factory()

    def _run_analyzers(
        self,
        sessions: List[Session],
        rule_files: List[RuleFileInfo],
        config: ReflectAnalyzerConfig,
    ) -> List[AnalysisResult]:
        """Run all enabled analyzers."""
        results = []
        for key in config.get_enabled():
            if key not in ANALYZER_CLASSES:
                logger.debug("Analyzer %s not yet implemented, skipping", key)
                continue
            try:
                analyzer = self._get_analyzer(key)
                result = analyzer.analyze(sessions, rule_files)
                results.append(result)
                logger.info("  %s: score=%.1f", key, result.score)
            except Exception as e:
                logger.warning("Analyzer %s failed: %s", key, e)
        return results

    def _persist(self, report: ReflectReport, tool: ToolType):
        """Persist report, results, insights, and trend points."""
        try:
            # Save report
            self.storage.save_report(report)

            # Save individual analysis results
            for r in report.results:
                self.storage.save_analysis_result(r, tool.value)

            # Save insights
            for insight in report.insights:
                self.storage.save_insight(
                    insight, tool.value,
                    report.period_start, report.period_end,
                )

            # Save trend points
            now = datetime.now(timezone.utc)
            for r in report.results:
                self.storage.save_trend_point(
                    r.analyzer_key, r.score, tool.value, now, "daily"
                )

            # Save session metadata
            # (we only save metadata, never raw turns)
            # Session metadata is saved during read for deduplication,
            # but we save aggregate here too
            self.storage.save_trend_point(
                "overall_score", report.overall_score, tool.value, now, "daily"
            )

        except Exception as e:
            logger.warning("Error persisting results: %s", e)
