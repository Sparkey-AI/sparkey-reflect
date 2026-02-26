"""
Context Management Analyzer

Evaluates how well users provide context to the AI across four dimensions:
- File Reference Rate (0-25): How often prompts include file paths/names
- Error Inclusion Rate (0-25): How often error context is provided when debugging
- Code Snippet Rate (0-25): How often relevant code is included
- Scope Clarity (0-25): How well the scope of the request is bounded
"""

import re
from typing import Dict, List, Optional

from sparkey_reflect.analyzers.base_analyzer import BaseReflectAnalyzer
from sparkey_reflect.core.models import (
    AnalysisResult,
    ConversationTurn,
    RuleFileInfo,
    Session,
    SessionType,
)

# Patterns that indicate scope-bounding language
SCOPE_PATTERNS = [
    r"\b(only|just|specifically|in this file|this function|this class|this method)\b",
    r"\b(don't (touch|change|modify)|leave .+ alone|keep .+ as is)\b",
    r"\b(scope|limited to|restrict|focused on|within)\b",
]

# Patterns indicating the user SHOULD have included error context but didn't
NEEDS_ERROR_CONTEXT = [
    r"\b(doesn't work|not working|broken|fails|failing|issue|problem)\b",
    r"\b(why (is|does|doesn't)|what's wrong|can't figure out)\b",
]


class ContextManagementAnalyzer(BaseReflectAnalyzer):
    """Analyzes how effectively users provide context to the AI."""

    def get_key(self) -> str:
        return "context_management"

    def get_name(self) -> str:
        return "Context Management"

    def analyze(
        self,
        sessions: List[Session],
        rule_files: Optional[List[RuleFileInfo]] = None,
    ) -> AnalysisResult:
        if not sessions:
            return AnalysisResult(
                analyzer_key=self.get_key(),
                analyzer_name=self.get_name(),
                score=0,
                session_count=0,
            )

        all_file_ref_rates = []
        all_error_rates = []
        all_code_rates = []
        all_scope_scores = []

        for session in sessions:
            user_turns = [t for t in session.turns if t.role == "user" and t.content]
            if not user_turns:
                continue

            # File reference rate
            with_files = sum(1 for t in user_turns if t.file_references)
            all_file_ref_rates.append(with_files / len(user_turns))

            # Error inclusion rate (only for sessions that look like debugging)
            error_rate = self._compute_error_inclusion(user_turns, session)
            if error_rate is not None:
                all_error_rates.append(error_rate)

            # Code snippet rate
            with_code = sum(1 for t in user_turns if t.has_code_snippet)
            all_code_rates.append(with_code / len(user_turns))

            # Scope clarity (per-turn scoring averaged)
            scope_scores = [self._score_turn_scope(t) for t in user_turns]
            all_scope_scores.append(sum(scope_scores) / len(scope_scores))

        avg = lambda vals: sum(vals) / len(vals) if vals else 0

        file_ref_rate = avg(all_file_ref_rates)
        error_inclusion = avg(all_error_rates) if all_error_rates else 0.5  # neutral if no debugging sessions
        code_snippet_rate = avg(all_code_rates)
        scope_clarity = avg(all_scope_scores)

        # Score each dimension 0-25
        file_score = self._score_file_refs(file_ref_rate)
        error_score = self._score_error_inclusion(error_inclusion)
        code_score = self._score_code_snippets(code_snippet_rate)
        scope_score = scope_clarity * 25  # already 0-1 from turn scoring

        overall = file_score + error_score + code_score + scope_score

        period_start = min((s.start_time for s in sessions if s.start_time), default=None)
        period_end = max((s.end_time for s in sessions if s.end_time), default=None)

        return AnalysisResult(
            analyzer_key=self.get_key(),
            analyzer_name=self.get_name(),
            score=max(0, min(100, overall)),
            metrics={
                "file_reference_rate": round(file_ref_rate, 3),
                "error_inclusion_rate": round(error_inclusion, 3),
                "code_snippet_rate": round(code_snippet_rate, 3),
                "scope_clarity": round(scope_clarity, 3),
                "sessions_analyzed": len(sessions),
                "debugging_sessions": len(all_error_rates),
            },
            insights=[],
            session_count=len(sessions),
            period_start=period_start,
            period_end=period_end,
        )

    # =========================================================================
    # Metric Computation
    # =========================================================================

    def _compute_error_inclusion(
        self, user_turns: List[ConversationTurn], session: Session
    ) -> Optional[float]:
        """
        Compute error inclusion rate for debugging-like sessions.
        Returns None if the session doesn't appear to involve debugging.
        """
        # Only measure this for sessions that look like debugging
        is_debug = session.session_type == SessionType.DEBUGGING
        if not is_debug:
            # Check if any turns mention issues without providing error context
            complaint_turns = [
                t for t in user_turns
                if any(re.search(p, t.content, re.IGNORECASE) for p in NEEDS_ERROR_CONTEXT)
            ]
            if not complaint_turns:
                return None  # not a debugging session
            # For complaint turns, how many include actual error text?
            with_error = sum(1 for t in complaint_turns if t.has_error_context)
            return with_error / len(complaint_turns) if complaint_turns else None

        # For debugging sessions, check all user turns
        with_error = sum(1 for t in user_turns if t.has_error_context)
        return with_error / len(user_turns)

    def _score_turn_scope(self, turn: ConversationTurn) -> float:
        """Score 0-1: How well a single turn defines scope."""
        text = turn.content
        score = 0.0

        # Has scope-bounding language
        for pattern in SCOPE_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                score += 0.3
                break

        # References specific files (narrows scope)
        if turn.file_references:
            score += 0.25

        # Short-to-medium length (not a vague one-liner, not a dump)
        word_count = len(text.split())
        if 15 <= word_count <= 200:
            score += 0.2
        elif 5 <= word_count < 15:
            score += 0.1

        # Mentions specific identifiers
        identifiers = re.findall(r'\b[a-z_][a-zA-Z0-9_]{2,}\b', text)
        if len(identifiers) >= 2:
            score += 0.15

        # Mentions constraints/boundaries
        constraints = r'\b(without|don\'t|must not|keep|preserve|maintain)\b'
        if re.search(constraints, text, re.IGNORECASE):
            score += 0.1

        return min(1.0, score)

    # =========================================================================
    # Scoring Dimensions (0-25 each)
    # =========================================================================

    def _score_file_refs(self, rate: float) -> float:
        """Score 0-25: File reference rate."""
        # 50%+ turns with file refs = excellent
        if rate >= 0.5:
            return 25
        if rate >= 0.3:
            return 20
        if rate >= 0.15:
            return 14
        if rate >= 0.05:
            return 8
        return 3

    def _score_error_inclusion(self, rate: float) -> float:
        """Score 0-25: Error inclusion rate in debugging turns."""
        if rate >= 0.7:
            return 25
        if rate >= 0.5:
            return 20
        if rate >= 0.3:
            return 14
        if rate >= 0.1:
            return 8
        return 4

    def _score_code_snippets(self, rate: float) -> float:
        """Score 0-25: Code snippet inclusion rate."""
        if rate >= 0.4:
            return 25
        if rate >= 0.25:
            return 20
        if rate >= 0.15:
            return 14
        if rate >= 0.05:
            return 8
        return 3

