"""
Rule File Quality Analyzer

Evaluates the quality of AI instruction/rule files across four dimensions:
- Completeness (0-25): Coverage of key areas (project context, examples, constraints, style)
- Specificity (0-25): Concrete, actionable instructions vs vague guidelines
- Actionability (0-25): Clear dos/don'ts, structured sections, imperative language
- Currency (0-25): File freshness, appropriate size, maintenance indicators
"""

import re
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

from sparkey_reflect.analyzers.base_analyzer import BaseReflectAnalyzer
from sparkey_reflect.core.models import (
    AnalysisResult,
    InsightCategory,
    InsightSeverity,
    ReflectInsight,
    RuleFileInfo,
    Session,
    ToolType,
)

# Primary instruction file types per tool
PRIMARY_RULE_FILES = {
    ToolType.CLAUDE_CODE: "claude_md",
    ToolType.CURSOR: "cursorrules",
    ToolType.COPILOT: "copilot_instructions",
}

# Recommended minimum word counts for primary instruction files
MIN_WORD_COUNT = 100
GOOD_WORD_COUNT = 300
EXCELLENT_WORD_COUNT = 800

# Specificity patterns (concrete instructions)
SPECIFICITY_PATTERNS = [
    r'\b(use|prefer|always use|never use)\s+\w+',  # "use snake_case"
    r'`[^`]+`',  # inline code references
    r'\b(version|v\d|python\s*\d|node\s*\d)',  # version mentions
    r'\b(import|from|require|include)\s+\w+',  # specific imports/packages
    r'\b(directory|folder|path)\s*[:/]\s*\S+',  # specific paths
]

# Actionability patterns (imperative language)
ACTIONABILITY_PATTERNS = [
    r'^[-*]\s',  # bullet points
    r'^\d+[\.\)]\s',  # numbered lists
    r'\b(DO|DON\'T|MUST|NEVER|ALWAYS|IMPORTANT|CRITICAL|NOTE)\b',  # caps emphasis
    r'\b(should|shall|must|require|forbidden|prohibited|mandatory)\b',  # modal verbs
]

# Currency: stale threshold
STALE_DAYS = 90


class RuleFileAnalyzer(BaseReflectAnalyzer):
    """Analyzes quality of AI instruction and rule files."""

    def get_key(self) -> str:
        return "rule_file"

    def get_name(self) -> str:
        return "Rule File Quality"

    def analyze(
        self,
        sessions: List[Session],
        rule_files: Optional[List[RuleFileInfo]] = None,
    ) -> AnalysisResult:
        if not rule_files:
            return AnalysisResult(
                analyzer_key=self.get_key(),
                analyzer_name=self.get_name(),
                score=0,
                session_count=len(sessions) if sessions else 0,
                insights=[ReflectInsight(
                    category=InsightCategory.RULE_FILE_QUALITY,
                    title="No rule files found",
                    severity=InsightSeverity.WARNING,
                    recommendation=(
                        "No instruction files were found. Create a CLAUDE.md, "
                        ".cursorrules, or .github/copilot-instructions.md to "
                        "give the AI project-specific context and guidelines."
                    ),
                    evidence="No rule files detected in workspace",
                    metric_key="rule_file_count",
                    metric_value=0,
                )],
            )

        existing_files = [rf for rf in rule_files if rf.exists]
        if not existing_files:
            return AnalysisResult(
                analyzer_key=self.get_key(),
                analyzer_name=self.get_name(),
                score=10,
                session_count=len(sessions) if sessions else 0,
                metrics={"existing_count": 0, "total_checked": len(rule_files)},
                insights=[ReflectInsight(
                    category=InsightCategory.RULE_FILE_QUALITY,
                    title="No rule files exist",
                    severity=InsightSeverity.WARNING,
                    recommendation=(
                        "None of the expected instruction files exist. "
                        "Creating even a minimal instruction file significantly "
                        "improves AI response quality."
                    ),
                    evidence=f"Checked {len(rule_files)} paths, none exist",
                    metric_key="rule_file_count",
                    metric_value=0,
                )],
            )

        # Determine primary file (most important instruction file)
        tool = existing_files[0].tool
        primary_type = PRIMARY_RULE_FILES.get(tool, "claude_md")
        primary = next(
            (rf for rf in existing_files if rf.file_type == primary_type), None
        )

        # Score each dimension across all existing files
        completeness = self._score_completeness(existing_files, primary)
        specificity = self._score_specificity(existing_files)
        actionability = self._score_actionability(existing_files)
        currency = self._score_currency(existing_files)

        overall = completeness + specificity + actionability + currency

        period_start = min((s.start_time for s in sessions if s.start_time), default=None) if sessions else None
        period_end = max((s.end_time for s in sessions if s.end_time), default=None) if sessions else None

        total_words = sum(rf.word_count for rf in existing_files)
        total_sections = sum(rf.section_count for rf in existing_files)

        return AnalysisResult(
            analyzer_key=self.get_key(),
            analyzer_name=self.get_name(),
            score=max(0, min(100, overall)),
            metrics={
                "existing_count": len(existing_files),
                "total_checked": len(rule_files),
                "total_words": total_words,
                "total_sections": total_sections,
                "has_primary": primary is not None,
                "completeness": round(completeness, 1),
                "specificity": round(specificity, 1),
                "actionability": round(actionability, 1),
                "currency": round(currency, 1),
            },
            insights=[],
            session_count=len(sessions) if sessions else 0,
            period_start=period_start,
            period_end=period_end,
        )

    # =========================================================================
    # Scoring Dimensions (0-25 each)
    # =========================================================================

    def _score_completeness(
        self, existing: List[RuleFileInfo], primary: Optional[RuleFileInfo]
    ) -> float:
        """Score 0-25: Coverage of key instruction areas."""
        score = 0.0

        # Primary file exists
        if primary:
            score += 5

            # Word count check
            if primary.word_count >= EXCELLENT_WORD_COUNT:
                score += 5
            elif primary.word_count >= GOOD_WORD_COUNT:
                score += 3
            elif primary.word_count >= MIN_WORD_COUNT:
                score += 1

        # Coverage flags (across all files)
        has = lambda attr: any(getattr(rf, attr, False) for rf in existing)
        if has("has_project_context"):
            score += 4
        if has("has_examples"):
            score += 4
        if has("has_constraints"):
            score += 4
        if has("has_style_guide"):
            score += 3

        return min(25, score)

    def _score_specificity(self, existing: List[RuleFileInfo]) -> float:
        """Score 0-25: Concrete, specific instructions."""
        if not existing:
            return 0

        all_content = " ".join(rf.raw_content or "" for rf in existing)
        if not all_content.strip():
            return 0

        score = 0.0
        total_lines = all_content.count("\n") + 1

        # Count specificity pattern matches
        match_count = 0
        for pattern in SPECIFICITY_PATTERNS:
            match_count += len(re.findall(pattern, all_content, re.IGNORECASE | re.MULTILINE))

        # Density of specific instructions
        density = match_count / max(total_lines, 1)
        if density >= 0.3:
            score += 12
        elif density >= 0.15:
            score += 8
        elif density >= 0.05:
            score += 4

        # Section count indicates structure
        total_sections = sum(rf.section_count for rf in existing)
        if total_sections >= 10:
            score += 8
        elif total_sections >= 5:
            score += 5
        elif total_sections >= 2:
            score += 3

        # Code examples (backtick blocks)
        code_blocks = len(re.findall(r'```', all_content))
        if code_blocks >= 6:
            score += 5
        elif code_blocks >= 2:
            score += 3
        elif code_blocks >= 1:
            score += 1

        return min(25, score)

    def _score_actionability(self, existing: List[RuleFileInfo]) -> float:
        """Score 0-25: Clear, actionable language and structure."""
        if not existing:
            return 0

        all_content = " ".join(rf.raw_content or "" for rf in existing)
        if not all_content.strip():
            return 0

        score = 0.0
        lines = all_content.split("\n")

        # Structured lists (bullets, numbers)
        list_lines = sum(
            1 for line in lines
            if re.match(r'^\s*[-*]\s', line) or re.match(r'^\s*\d+[\.\)]\s', line)
        )
        list_ratio = list_lines / max(len(lines), 1)
        if list_ratio >= 0.2:
            score += 8
        elif list_ratio >= 0.1:
            score += 5
        elif list_lines >= 3:
            score += 3

        # Imperative / emphasis language
        emphasis_count = 0
        for pattern in ACTIONABILITY_PATTERNS[2:]:  # caps + modals
            emphasis_count += len(re.findall(pattern, all_content, re.MULTILINE))
        if emphasis_count >= 10:
            score += 8
        elif emphasis_count >= 5:
            score += 5
        elif emphasis_count >= 2:
            score += 3

        # Do/Don't pairs (clear guidance)
        dos = len(re.findall(r'\b(do|always|prefer|use)\b', all_content, re.IGNORECASE))
        donts = len(re.findall(r"\b(don't|never|avoid|do not)\b", all_content, re.IGNORECASE))
        if dos >= 3 and donts >= 2:
            score += 6
        elif dos >= 2 or donts >= 1:
            score += 3

        # Headers create scannable structure
        header_count = sum(rf.section_count for rf in existing)
        if header_count >= 8:
            score += 3
        elif header_count >= 3:
            score += 2

        return min(25, score)

    def _score_currency(self, existing: List[RuleFileInfo]) -> float:
        """Score 0-25: Freshness and maintenance of rule files."""
        if not existing:
            return 0

        score = 0.0
        now = datetime.now(timezone.utc)

        # Check primary file freshness
        dates = [rf.last_modified for rf in existing if rf.last_modified]
        if not dates:
            return 12  # neutral - can't determine

        most_recent = max(dates)
        days_since = (now - most_recent).days

        if days_since <= 7:
            score += 12
        elif days_since <= 30:
            score += 10
        elif days_since <= STALE_DAYS:
            score += 6
        else:
            score += 2

        # Appropriate size (not too small, not bloated)
        total_words = sum(rf.word_count for rf in existing)
        if GOOD_WORD_COUNT <= total_words <= 5000:
            score += 8
        elif MIN_WORD_COUNT <= total_words < GOOD_WORD_COUNT:
            score += 5
        elif total_words > 5000:
            score += 4  # overly large might be unfocused
        else:
            score += 2

        # Multiple files = better maintenance (settings, memory, etc.)
        if len(existing) >= 4:
            score += 5
        elif len(existing) >= 2:
            score += 3
        else:
            score += 1

        return min(25, score)

