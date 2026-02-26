"""
LLM Insight Generator

Generates personalized coaching insights by sending analyzer scores and
cleaned conversation data to Claude.

Authentication: Claude Code CLI (`claude --print`) using existing OAuth session.
"""

import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from sparkey_reflect.core.models import (
    AnalysisResult,
    InsightCategory,
    InsightSeverity,
    ReflectInsight,
    RuleFileInfo,
    Session,
    TrendDirection,
)
from sparkey_reflect.insights.conversation_extractor import ConversationExtractor

logger = logging.getLogger(__name__)

# Path to the system prompt markdown file
_PROMPT_DIR = Path(__file__).parent / "prompts"
_SYSTEM_PROMPT_PATH = _PROMPT_DIR / "reflect_system_prompt.md"

# Category string -> enum mapping
_CATEGORY_MAP = {
    "prompt_engineering": InsightCategory.PROMPT_ENGINEERING,
    "conversation_flow": InsightCategory.CONVERSATION_FLOW,
    "context_management": InsightCategory.CONTEXT_MANAGEMENT,
    "tool_mastery": InsightCategory.TOOL_MASTERY,
    "rule_file_quality": InsightCategory.RULE_FILE_QUALITY,
    "session_habits": InsightCategory.SESSION_HABITS,
    "outcome_quality": InsightCategory.OUTCOME_QUALITY,
    "completion_usage": InsightCategory.COMPLETION_USAGE,
}

_SEVERITY_MAP = {
    "info": InsightSeverity.INFO,
    "suggestion": InsightSeverity.SUGGESTION,
    "warning": InsightSeverity.WARNING,
    "critical": InsightSeverity.CRITICAL,
}


def _find_claude_cli() -> Optional[str]:
    """Find the Claude Code CLI binary."""
    home_path = Path.home() / ".claude" / "local" / "claude"
    if home_path.exists():
        return str(home_path)
    return shutil.which("claude")


class LLMInsightGenerator:
    """Generates insights by sending analyzer scores and conversation data to Claude."""

    def __init__(
        self,
        model: Optional[str] = None,
        max_output_tokens: int = 16384,
        context_window_limit: int = 180_000,
    ):
        from sparkey_reflect.config.defaults import DEFAULT_MODEL

        self.model = model or os.environ.get("REFLECT_MODEL") or DEFAULT_MODEL
        self.max_output_tokens = max_output_tokens
        self.context_window_limit = context_window_limit
        self.extractor = ConversationExtractor()
        self._system_prompt: Optional[str] = None

    def generate_insights(
        self,
        results: List[AnalysisResult],
        sessions: List[Session],
        rule_files: Optional[List[RuleFileInfo]] = None,
        trends: Optional[Dict[str, TrendDirection]] = None,
    ) -> Dict[str, Any]:
        """
        Generate LLM-powered insights from analyzer results and conversation data.

        Uses Claude Code CLI with existing OAuth session.

        Returns a dict with keys: overall_assessment, insights, learning_path,
        rule_file_suggestions.
        """
        system_prompt = self._load_system_prompt()
        user_prompt = self._build_user_prompt(results, sessions, rule_files, trends)

        raw_text = self._call_via_cli(system_prompt, user_prompt)
        if raw_text is not None:
            return self._parse_response(raw_text)

        return self._fallback_response(
            "Claude Code CLI not available. Install Claude Code and run 'claude auth login'."
        )

    def parse_insights(self, llm_data: Dict[str, Any]) -> List[ReflectInsight]:
        """Convert parsed LLM response into ReflectInsight objects."""
        insights = []
        for item in llm_data.get("insights", []):
            category = _CATEGORY_MAP.get(
                item.get("category", ""), InsightCategory.PROMPT_ENGINEERING
            )
            severity = _SEVERITY_MAP.get(
                item.get("severity", ""), InsightSeverity.SUGGESTION
            )
            insights.append(ReflectInsight(
                category=category,
                title=item.get("title", "Insight"),
                severity=severity,
                recommendation=item.get("recommendation", ""),
                evidence=item.get("evidence", ""),
            ))
        return insights

    # =========================================================================
    # LLM Call Backend
    # =========================================================================

    def _call_via_cli(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        """Call Claude via the Claude Code CLI (uses existing OAuth session)."""
        claude_bin = _find_claude_cli()
        if not claude_bin:
            logger.debug("Claude Code CLI not found")
            return None

        logger.info("Using Claude Code CLI for LLM insights (OAuth)")
        timeout_secs = 300

        try:
            # Remove CLAUDECODE env var so subprocess can run inside a Claude Code session
            env = os.environ.copy()
            env.pop("CLAUDECODE", None)

            result = subprocess.run(
                [
                    claude_bin,
                    "--print",
                    "--output-format", "json",
                    "--system-prompt", system_prompt,
                    "--no-session-persistence",
                    "--model", self.model,
                    "--allowedTools", "",
                ],
                input=user_prompt,
                capture_output=True,
                text=True,
                timeout=timeout_secs,
                env=env,
            )
            if result.returncode != 0:
                stderr = result.stderr.strip()
                logger.warning("Claude CLI exited with code %d: %s", result.returncode, stderr[:500])
                return None

            raw_text = result.stdout.strip()
            if not raw_text:
                logger.warning("Claude CLI returned empty output")
                return None

            logger.info("LLM insights generated via Claude Code CLI")
            return raw_text
        except subprocess.TimeoutExpired:
            logger.warning("Claude CLI timed out after %ds", timeout_secs)
            return None
        except (FileNotFoundError, OSError) as e:
            logger.warning("Failed to run Claude CLI: %s", e)
            return None

    # =========================================================================
    # Prompt Construction
    # =========================================================================

    def _load_system_prompt(self) -> str:
        """Load the system prompt from the markdown file."""
        if self._system_prompt is None:
            try:
                self._system_prompt = _SYSTEM_PROMPT_PATH.read_text()
            except FileNotFoundError:
                logger.warning("System prompt file not found at %s", _SYSTEM_PROMPT_PATH)
                self._system_prompt = "You are an AI coding advisor. Analyze the developer's AI usage and provide actionable insights as JSON."
        return self._system_prompt

    def _build_user_prompt(
        self,
        results: List[AnalysisResult],
        sessions: List[Session],
        rule_files: Optional[List[RuleFileInfo]],
        trends: Optional[Dict[str, TrendDirection]],
    ) -> str:
        """Build the user prompt with scores, conversations, rules, and trends."""
        parts = []

        # Section 1: Analyzer scores
        parts.append("## Analyzer Scores\n")
        for r in results:
            parts.append(f"### {r.analyzer_name} (key: {r.analyzer_key})")
            parts.append(f"Score: {r.score:.1f}/100")
            if r.metrics:
                metrics_str = ", ".join(
                    f"{k}={v}" for k, v in r.metrics.items()
                    if not isinstance(v, dict)
                )
                if metrics_str:
                    parts.append(f"Metrics: {metrics_str}")
                # Include nested dicts (like task_type_distribution)
                for k, v in r.metrics.items():
                    if isinstance(v, dict):
                        parts.append(f"{k}: {json.dumps(v)}")
            parts.append("")

        # Section 2: Trends
        if trends:
            parts.append("## Trends\n")
            for key, direction in trends.items():
                parts.append(f"- {key}: {direction.value}")
            parts.append("")

        # Section 3: Rule files
        if rule_files:
            parts.append("## Rule Files\n")
            for rf in rule_files:
                status = "exists" if rf.exists else "MISSING"
                parts.append(f"### {rf.file_type} ({status})")
                if rf.exists:
                    parts.append(f"Words: {rf.word_count}, Sections: {rf.section_count}")
                    flags = []
                    if rf.has_examples:
                        flags.append("has_examples")
                    if rf.has_constraints:
                        flags.append("has_constraints")
                    if rf.has_project_context:
                        flags.append("has_project_context")
                    if rf.has_style_guide:
                        flags.append("has_style_guide")
                    if flags:
                        parts.append(f"Features: {', '.join(flags)}")
                    if rf.raw_content:
                        content_preview = rf.raw_content[:3000]
                        if len(rf.raw_content) > 3000:
                            content_preview += f"\n... (truncated, {rf.word_count} words total)"
                        parts.append(f"Content:\n```\n{content_preview}\n```")
                parts.append("")

        # Section 4: Conversations
        extracted = self.extractor.extract(sessions)
        estimated_tokens = self.extractor.estimate_tokens(extracted)

        if estimated_tokens > self.context_window_limit:
            extracted = self._trim_sessions(extracted, sessions)

        if extracted:
            parts.append("## Conversation History\n")
            parts.append(self.extractor.to_prompt_text(extracted))

        return "\n".join(parts)

    def _trim_sessions(
        self,
        extracted,
        sessions: List[Session],
    ) -> list:
        """Trim extracted sessions to fit within context window.

        Keeps the most recent sessions verbatim and drops older ones.
        """
        if not extracted:
            return extracted

        sorted_extracted = sorted(
            extracted,
            key=lambda es: es.timestamp or "",
        )

        keep_count = max(5, len(sorted_extracted) // 3)
        trimmed = sorted_extracted[-keep_count:]

        estimated = self.extractor.estimate_tokens(trimmed)
        while estimated > self.context_window_limit and len(trimmed) > 5:
            trimmed = trimmed[1:]  # drop oldest
            estimated = self.extractor.estimate_tokens(trimmed)

        return trimmed

    # =========================================================================
    # Response Parsing
    # =========================================================================

    def _parse_response(self, raw_text: str) -> Dict[str, Any]:
        """Parse the LLM JSON response, with fallback for malformed output."""
        text = raw_text.strip()

        # Handle markdown code blocks
        if text.startswith("```"):
            lines = text.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines)

        try:
            data = json.loads(text)
            if isinstance(data, dict) and "insights" in data:
                return data
            logger.warning("LLM response missing 'insights' key, wrapping as raw")
            return self._wrap_raw_response(raw_text)
        except json.JSONDecodeError:
            logger.warning("Failed to parse LLM response as JSON")
            return self._wrap_raw_response(raw_text)

    def _wrap_raw_response(self, raw_text: str) -> Dict[str, Any]:
        """Wrap unparseable LLM output as a single insight."""
        return {
            "overall_assessment": "Analysis completed (raw output).",
            "insights": [
                {
                    "category": "prompt_engineering",
                    "title": "AI Analysis",
                    "severity": "info",
                    "recommendation": raw_text[:2000],
                    "evidence": "Raw LLM output (JSON parsing failed)",
                }
            ],
            "learning_path": [],
            "rule_file_suggestions": None,
        }

    def _fallback_response(self, reason: str) -> Dict[str, Any]:
        """Return a minimal response when the LLM call can't be made."""
        return {
            "overall_assessment": None,
            "insights": [
                {
                    "category": "prompt_engineering",
                    "title": "LLM insights unavailable",
                    "severity": "info",
                    "recommendation": reason,
                    "evidence": "LLM generation was skipped",
                }
            ],
            "learning_path": [],
            "rule_file_suggestions": None,
        }
