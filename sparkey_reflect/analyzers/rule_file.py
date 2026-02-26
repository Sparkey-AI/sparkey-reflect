"""
Rule File Quality Analyzer

Evaluates the quality of AI instruction/rule files across five dimensions
using smooth scoring curves grounded in industry benchmarks:
- Completeness (w=0.25): Coverage of key areas + primary file bonus
- Specificity (w=0.25): Pattern density of concrete instructions
- Actionability (w=0.20): Clear dos/don'ts, imperative language density
- Currency (w=0.15): Freshness (DORA: stale docs = risk)
- Ecosystem Coverage (w=0.15): Multiple instruction surfaces (NEW)

Benchmarks: DORA (stale documentation increases risk), broader ecosystem
coverage (CLAUDE.md, .cursorrules, .mcp.json, memory files) = more
comprehensive AI guidance.
"""

import re
from datetime import datetime, timezone
from typing import List, Optional

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
from sparkey_reflect.core.scoring import count_score, sigmoid, weighted_sum

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
    r'\b(use|prefer|always use|never use)\s+\w+',
    r'`[^`]+`',
    r'\b(version|v\d|python\s*\d|node\s*\d)',
    r'\b(import|from|require|include)\s+\w+',
    r'\b(directory|folder|path)\s*[:/]\s*\S+',
]

# Actionability patterns (imperative language)
ACTIONABILITY_PATTERNS = [
    r'^[-*]\s',
    r'^\d+[\.\)]\s',
    r'\b(DO|DON\'T|MUST|NEVER|ALWAYS|IMPORTANT|CRITICAL|NOTE)\b',
    r'\b(should|shall|must|require|forbidden|prohibited|mandatory)\b',
]

# Currency: stale threshold
STALE_DAYS = 90

# Distinct rule file types for ecosystem coverage
ECOSYSTEM_FILE_TYPES = {
    "claude_md", "cursorrules", "copilot_instructions",
    "mcp_config", "claude_user_mcp",
    "claude_settings", "memory",
}


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

        # Determine primary file
        tool = existing_files[0].tool
        primary_type = PRIMARY_RULE_FILES.get(tool, "claude_md")
        primary = next(
            (rf for rf in existing_files if rf.file_type == primary_type), None
        )

        # Compute raw signals for each dimension
        completeness_signals = self._count_completeness_signals(existing_files, primary)
        specificity_density = self._compute_specificity_density(existing_files)
        actionability_signals = self._count_actionability_signals(existing_files)
        days_since_update = self._compute_days_since_update(existing_files)
        ecosystem_count = self._count_ecosystem_files(existing_files)

        # Smooth scoring: each dimension 0-1
        completeness_dim = sigmoid(completeness_signals, 3, 0.8)
        # Add primary file bonus
        if primary:
            completeness_dim = min(1.0, completeness_dim + 0.15)

        specificity_dim = sigmoid(specificity_density, 0.15, 8)
        actionability_dim = sigmoid(actionability_signals, 5, 0.5)

        if days_since_update is None:
            currency_dim = 0.5  # neutral - can't determine
        else:
            currency_dim = 1 - sigmoid(days_since_update, 45, 0.05)

        ecosystem_dim = count_score(ecosystem_count, [
            (1, 0.3), (2, 0.5), (3, 0.7), (4, 0.85), (5, 1.0),
        ])

        overall = weighted_sum([
            (completeness_dim, 0.25),
            (specificity_dim, 0.25),
            (actionability_dim, 0.20),
            (currency_dim, 0.15),
            (ecosystem_dim, 0.15),
        ])

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
                "completeness": round(completeness_dim, 3),
                "specificity": round(specificity_dim, 3),
                "actionability": round(actionability_dim, 3),
                "currency": round(currency_dim, 3),
                "ecosystem_coverage": ecosystem_count,
            },
            insights=[],
            session_count=len(sessions) if sessions else 0,
            period_start=period_start,
            period_end=period_end,
        )

    # =========================================================================
    # Signal Computation
    # =========================================================================

    def _count_completeness_signals(
        self, existing: List[RuleFileInfo], primary: Optional[RuleFileInfo]
    ) -> float:
        """Count completeness coverage flags (0-6+)."""
        signals = 0.0

        if primary:
            if primary.word_count >= EXCELLENT_WORD_COUNT:
                signals += 1.5
            elif primary.word_count >= GOOD_WORD_COUNT:
                signals += 1
            elif primary.word_count >= MIN_WORD_COUNT:
                signals += 0.5

        has = lambda attr: any(getattr(rf, attr, False) for rf in existing)
        if has("has_project_context"):
            signals += 1
        if has("has_examples"):
            signals += 1
        if has("has_constraints"):
            signals += 1
        if has("has_style_guide"):
            signals += 0.8

        return signals

    def _compute_specificity_density(self, existing: List[RuleFileInfo]) -> float:
        """Compute density of specificity patterns (matches per line)."""
        all_content = " ".join(rf.raw_content or "" for rf in existing)
        if not all_content.strip():
            return 0.0

        total_lines = max(all_content.count("\n") + 1, 1)
        match_count = 0
        for pattern in SPECIFICITY_PATTERNS:
            match_count += len(re.findall(pattern, all_content, re.IGNORECASE | re.MULTILINE))

        return match_count / total_lines

    def _count_actionability_signals(self, existing: List[RuleFileInfo]) -> float:
        """Count actionability signals across all files."""
        all_content = " ".join(rf.raw_content or "" for rf in existing)
        if not all_content.strip():
            return 0.0

        signals = 0.0
        lines = all_content.split("\n")

        # Structured lists
        list_lines = sum(
            1 for line in lines
            if re.match(r'^\s*[-*]\s', line) or re.match(r'^\s*\d+[\.\)]\s', line)
        )
        list_ratio = list_lines / max(len(lines), 1)
        signals += list_ratio * 10  # scale up

        # Emphasis language
        for pattern in ACTIONABILITY_PATTERNS[2:]:
            signals += len(re.findall(pattern, all_content, re.MULTILINE)) * 0.3

        # Do/Don't pairs
        dos = len(re.findall(r'\b(do|always|prefer|use)\b', all_content, re.IGNORECASE))
        donts = len(re.findall(r"\b(don't|never|avoid|do not)\b", all_content, re.IGNORECASE))
        signals += min(dos, 5) * 0.3
        signals += min(donts, 5) * 0.3

        return signals

    def _compute_days_since_update(self, existing: List[RuleFileInfo]) -> Optional[float]:
        """Compute days since most recent update. Returns None if unknown."""
        dates = [rf.last_modified for rf in existing if rf.last_modified]
        if not dates:
            return None
        most_recent = max(dates)
        return (datetime.now(timezone.utc) - most_recent).days

    def _count_ecosystem_files(self, existing: List[RuleFileInfo]) -> int:
        """Count distinct rule file types for ecosystem coverage."""
        types_found = set()
        for rf in existing:
            if rf.file_type in ECOSYSTEM_FILE_TYPES:
                types_found.add(rf.file_type)
            else:
                types_found.add(rf.file_type)
        return len(types_found)
