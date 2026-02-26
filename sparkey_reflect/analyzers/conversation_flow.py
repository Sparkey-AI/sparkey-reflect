"""
Conversation Flow Analyzer

Analyzes the dynamics of AI conversations:
- Turns to resolution: How many turns before the task is done
- Correction rate: How often the user has to correct the AI
- Context loss rate: How often context is re-stated
- First response acceptance: How often the first AI response is accepted
"""

import re
from typing import List, Optional

from sparkey_reflect.analyzers.base_analyzer import BaseReflectAnalyzer
from sparkey_reflect.core.models import (
    AnalysisResult,
    RuleFileInfo,
    Session,
)

# Patterns indicating the user is correcting the AI
CORRECTION_PATTERNS = [
    r"\b(no|wrong|incorrect|that's not|not what I|I said|I meant|instead)\b",
    r"\b(try again|redo|undo|revert|go back|start over)\b",
    r"\b(actually|wait|hold on|scratch that|never mind)\b",
]

# Patterns indicating context is being re-stated
CONTEXT_RESTATEMENT_PATTERNS = [
    r"\b(as I (said|mentioned)|like I said|remember|I already|again)\b",
    r"\b(the file I mentioned|the error (I showed|from before))\b",
    r"\b(same (file|function|error|issue)|still)\b",
]

# Patterns indicating task completion
COMPLETION_PATTERNS = [
    r"\b(thanks|perfect|great|looks good|that works|exactly|done)\b",
    r"\b(ship it|lgtm|merge|approved|nice)\b",
]

# Patterns indicating dissatisfaction / follow-up needed
FOLLOW_UP_PATTERNS = [
    r"\b(but|however|also|what about|one more|can you also)\b",
    r"\b(close but|almost|not quite|partially)\b",
]


class ConversationFlowAnalyzer(BaseReflectAnalyzer):
    """Analyzes conversation dynamics and flow efficiency."""

    def get_key(self) -> str:
        return "conversation_flow"

    def get_name(self) -> str:
        return "Conversation Flow"

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

        all_turns_to_resolution = []
        all_correction_rates = []
        all_context_loss_rates = []
        all_first_acceptance = []

        for session in sessions:
            user_turns = [t for t in session.turns if t.role == "user" and t.content]
            if not user_turns:
                continue

            # Turns to resolution (user turns per session)
            all_turns_to_resolution.append(len(user_turns))

            # Correction rate
            corrections = sum(
                1 for t in user_turns
                if any(re.search(p, t.content, re.IGNORECASE) for p in CORRECTION_PATTERNS)
            )
            rate = corrections / len(user_turns) if user_turns else 0
            all_correction_rates.append(rate)

            # Context loss / restatement rate
            restatements = sum(
                1 for t in user_turns
                if any(re.search(p, t.content, re.IGNORECASE) for p in CONTEXT_RESTATEMENT_PATTERNS)
            )
            restate_rate = restatements / len(user_turns) if user_turns else 0
            all_context_loss_rates.append(restate_rate)

            # First response acceptance
            if len(user_turns) >= 2:
                # Check if the second user message is a completion/acceptance
                second_msg = user_turns[1].content
                is_acceptance = any(
                    re.search(p, second_msg, re.IGNORECASE) for p in COMPLETION_PATTERNS
                )
                is_followup = any(
                    re.search(p, second_msg, re.IGNORECASE) for p in FOLLOW_UP_PATTERNS
                )
                is_correction = any(
                    re.search(p, second_msg, re.IGNORECASE) for p in CORRECTION_PATTERNS
                )

                if is_acceptance and not is_correction:
                    all_first_acceptance.append(1.0)
                elif is_followup:
                    all_first_acceptance.append(0.5)
                else:
                    all_first_acceptance.append(0.0)
            elif len(user_turns) == 1:
                # Single-turn session: accepted first response implicitly
                all_first_acceptance.append(1.0)

        avg = lambda vals: sum(vals) / len(vals) if vals else 0

        avg_turns = avg(all_turns_to_resolution)
        avg_correction = avg(all_correction_rates)
        avg_context_loss = avg(all_context_loss_rates)
        avg_first_accept = avg(all_first_acceptance)

        # Score calculation (0-100)
        # Fewer turns = better (ideal: 1-3 user turns)
        turns_score = max(0, 25 - max(0, (avg_turns - 2) * 4))
        # Lower correction rate = better
        correction_score = max(0, 25 * (1 - avg_correction * 3))
        # Lower context loss = better
        context_score = max(0, 25 * (1 - avg_context_loss * 3))
        # Higher first acceptance = better
        acceptance_score = avg_first_accept * 25

        overall = turns_score + correction_score + context_score + acceptance_score

        period_start = min((s.start_time for s in sessions if s.start_time), default=None)
        period_end = max((s.end_time for s in sessions if s.end_time), default=None)

        return AnalysisResult(
            analyzer_key=self.get_key(),
            analyzer_name=self.get_name(),
            score=max(0, min(100, overall)),
            metrics={
                "avg_turns_to_resolution": round(avg_turns, 2),
                "correction_rate": round(avg_correction, 3),
                "context_loss_rate": round(avg_context_loss, 3),
                "first_response_acceptance": round(avg_first_accept, 3),
                "sessions_analyzed": len(sessions),
            },
            insights=[],
            session_count=len(sessions),
            period_start=period_start,
            period_end=period_end,
        )

