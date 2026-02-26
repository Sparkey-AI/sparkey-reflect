"""
Context Management Analyzer

Evaluates how well users provide context to the AI across five dimensions
using smooth scoring curves grounded in industry benchmarks:
- File Reference Rate (w=0.25): How often prompts include file paths/names
- Error Inclusion Rate (w=0.20): Error context provided when debugging (DORA: faster MTTR)
- Code Snippet Rate (w=0.20): How often relevant code is included (METR)
- Scope Clarity (w=0.15): How well scope is bounded (SPACE: bounded tasks = faster delivery)
- Context Window Efficiency (w=0.20): Optimal context budget utilization (NEW)

Benchmarks: Grounding reduces hallucination, DORA (error context -> faster MTTR),
METR (code context -> better completions), SPACE (bounded tasks = faster delivery).
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
from sparkey_reflect.core.scoring import bell, sigmoid, weighted_sum

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

# Approximate context window sizes by model (tokens)
DEFAULT_CONTEXT_WINDOW = 200_000


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
        all_ctx_efficiency = []

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

            # Context window efficiency
            ctx_eff = self._compute_context_window_efficiency(session)
            all_ctx_efficiency.append(ctx_eff)

        avg = lambda vals: sum(vals) / len(vals) if vals else 0

        file_ref_rate = avg(all_file_ref_rates)
        error_inclusion = avg(all_error_rates) if all_error_rates else 0.5
        code_snippet_rate = avg(all_code_rates)
        scope_clarity = avg(all_scope_scores)
        ctx_efficiency = avg(all_ctx_efficiency)

        # Smooth scoring: each dimension 0-1
        file_dim = sigmoid(file_ref_rate, 0.25, 8)
        error_dim = sigmoid(error_inclusion, 0.4, 5)
        code_dim = sigmoid(code_snippet_rate, 0.2, 6)
        scope_dim = sigmoid(scope_clarity, 0.4, 4)
        ctx_eff_dim = bell(ctx_efficiency, 0.4, 0.25)

        overall = weighted_sum([
            (file_dim, 0.25),
            (error_dim, 0.20),
            (code_dim, 0.20),
            (scope_dim, 0.15),
            (ctx_eff_dim, 0.20),
        ])

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
                "context_window_efficiency": round(ctx_efficiency, 3),
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
        is_debug = session.session_type == SessionType.DEBUGGING
        if not is_debug:
            complaint_turns = [
                t for t in user_turns
                if any(re.search(p, t.content, re.IGNORECASE) for p in NEEDS_ERROR_CONTEXT)
            ]
            if not complaint_turns:
                return None
            with_error = sum(1 for t in complaint_turns if t.has_error_context)
            return with_error / len(complaint_turns) if complaint_turns else None

        with_error = sum(1 for t in user_turns if t.has_error_context)
        return with_error / len(user_turns)

    def _score_turn_scope(self, turn: ConversationTurn) -> float:
        """Score 0-1: How well a single turn defines scope."""
        text = turn.content
        score = 0.0

        for pattern in SCOPE_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                score += 0.3
                break

        if turn.file_references:
            score += 0.25

        word_count = len(text.split())
        if 15 <= word_count <= 200:
            score += 0.2
        elif 5 <= word_count < 15:
            score += 0.1

        identifiers = re.findall(r'\b[a-z_][a-zA-Z0-9_]{2,}\b', text)
        if len(identifiers) >= 2:
            score += 0.15

        constraints = r'\b(without|don\'t|must not|keep|preserve|maintain)\b'
        if re.search(constraints, text, re.IGNORECASE):
            score += 0.1

        return min(1.0, score)

    def _compute_context_window_efficiency(self, session: Session) -> float:
        """Estimate context utilization as fraction of model's context window.

        Optimal: 30-50% of context budget (bell curve centered at 0.4).
        Too low = underutilizing the model. Too high = cramming/overflow risk.
        """
        total_tokens = session.total_tokens
        if total_tokens <= 0:
            return 0.3  # neutral default

        utilization = total_tokens / DEFAULT_CONTEXT_WINDOW
        return min(1.0, utilization)
