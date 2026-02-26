"""
Prompt Quality Analyzer

Evaluates the quality of user prompts across five dimensions using smooth
scoring curves grounded in industry benchmarks:
- Specificity (w=0.25): How precise and targeted the prompt is
- Context Richness (w=0.25): File refs, error context, code snippets provided
- Clarity (w=0.20): Structure, actionable language, unambiguous intent
- Efficiency (w=0.15): Token economy, optimal prompt length (~40-150 words)
- Chain of Thought (w=0.15): Structured reasoning and acceptance criteria (NEW)

Benchmarks: GitClear (specific prompts -> 40% less churn), METR (context-rich
prompts -> 2x task completion), DevEx (clear intent reduces iteration).
"""

import re
from typing import List, Optional

from sparkey_reflect.analyzers.base_analyzer import BaseReflectAnalyzer
from sparkey_reflect.core.models import (
    AnalysisResult,
    ConversationTurn,
    RuleFileInfo,
    Session,
)
from sparkey_reflect.core.scoring import bell, sigmoid, weighted_sum


class PromptQualityAnalyzer(BaseReflectAnalyzer):
    """Analyzes the quality of user prompts."""

    def get_key(self) -> str:
        return "prompt_quality"

    def get_name(self) -> str:
        return "Prompt Quality"

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

        all_specificity = []
        all_context = []
        all_clarity = []
        all_efficiency = []
        all_cot = []

        for session in sessions:
            user_turns = [t for t in session.turns if t.role == "user" and t.content]
            for turn in user_turns:
                all_specificity.append(self._score_specificity(turn))
                all_context.append(self._score_context_richness(turn))
                all_clarity.append(self._score_clarity(turn))
                all_efficiency.append(self._score_efficiency(turn))
                all_cot.append(self._score_chain_of_thought(turn))

        if not all_specificity:
            return AnalysisResult(
                analyzer_key=self.get_key(),
                analyzer_name=self.get_name(),
                score=0,
                session_count=len(sessions),
            )

        avg = lambda vals: sum(vals) / len(vals) if vals else 0
        specificity = avg(all_specificity)
        context = avg(all_context)
        clarity = avg(all_clarity)
        efficiency = avg(all_efficiency)
        chain_of_thought = avg(all_cot)

        overall = weighted_sum([
            (specificity, 0.25),
            (context, 0.25),
            (clarity, 0.20),
            (efficiency, 0.15),
            (chain_of_thought, 0.15),
        ])

        period_start = min((s.start_time for s in sessions if s.start_time), default=None)
        period_end = max((s.end_time for s in sessions if s.end_time), default=None)

        return AnalysisResult(
            analyzer_key=self.get_key(),
            analyzer_name=self.get_name(),
            score=max(0, min(100, overall)),
            metrics={
                "specificity": round(specificity, 3),
                "context_richness": round(context, 3),
                "clarity": round(clarity, 3),
                "efficiency": round(efficiency, 3),
                "chain_of_thought": round(chain_of_thought, 3),
                "prompts_analyzed": len(all_specificity),
            },
            insights=[],
            session_count=len(sessions),
            period_start=period_start,
            period_end=period_end,
        )

    # =========================================================================
    # Scoring Dimensions (each returns 0.0-1.0)
    # =========================================================================

    def _score_specificity(self, turn: ConversationTurn) -> float:
        """Score 0-1: How specific and targeted the prompt is.

        Benchmark: GitClear finds specific prompts produce 40% less code churn.
        """
        text = turn.content
        word_count = len(text.split())

        if word_count < 5:
            return 0.1

        signals = 0

        # File references
        if turn.file_references:
            signals += 1

        # Function/class/variable names (camelCase, snake_case, PascalCase)
        identifiers = re.findall(r'\b[a-z_][a-zA-Z0-9_]{2,}\b', text)
        if len(identifiers) >= 2:
            signals += 1.5
        elif identifiers:
            signals += 0.8

        # Line numbers or specific locations
        if re.search(r'line\s+\d+|:\d+', text):
            signals += 1

        # Specific action verbs
        specific_verbs = r'\b(add|remove|rename|change|fix|update|create|delete|move|extract|refactor|implement|replace|modify)\b'
        if re.search(specific_verbs, text, re.IGNORECASE):
            signals += 1

        # Technology mentions
        tech_pattern = r'\b(React|Python|TypeScript|FastAPI|SQLAlchemy|Postgres|Redis|Docker|AWS|jest|pytest)\b'
        if re.search(tech_pattern, text, re.IGNORECASE):
            signals += 0.8

        # Well-scoped length (20-200 words)
        if 20 <= word_count <= 200:
            signals += 1

        # Penalize vague language
        vague_patterns = r'\b(help me|can you|please|something|anything|stuff|things|somehow)\b'
        vague_count = len(re.findall(vague_patterns, text, re.IGNORECASE))
        signals -= vague_count * 0.5

        signals = max(0, signals)
        return sigmoid(signals, 4, 0.8)

    def _score_context_richness(self, turn: ConversationTurn) -> float:
        """Score 0-1: How much useful context the prompt includes.

        Benchmark: METR finds context-rich prompts achieve 2x task completion.
        """
        signals = 0

        # File references (multi-file is more context)
        ref_count = len(turn.file_references)
        if ref_count >= 3:
            signals += 1.5
        elif ref_count >= 1:
            signals += 1

        # Error context
        if turn.has_error_context:
            signals += 1

        # Code snippets
        if turn.has_code_snippet:
            signals += 1

        # Expected behavior described
        expected = r'\b(should|expect|want|need|goal|output|result|return)\b'
        if re.search(expected, turn.content, re.IGNORECASE):
            signals += 0.8

        # Constraints mentioned
        constraints = r'\b(without|don\'t|must not|keep|preserve|maintain|backward.?compat)\b'
        if re.search(constraints, turn.content, re.IGNORECASE):
            signals += 0.7

        return sigmoid(signals, 3, 0.7)

    def _score_clarity(self, turn: ConversationTurn) -> float:
        """Score 0-1: Structural clarity and unambiguous intent.

        Benchmark: DevEx research shows clear intent reduces iteration cycles.
        """
        text = turn.content
        signals = 0

        # Structured with numbered steps or bullet points
        if re.search(r'^\s*(\d+[\.\)]\s|[-*]\s)', text, re.MULTILINE):
            signals += 1.2

        # Clear imperative sentence
        if re.match(r'^[A-Z][a-z]', text.strip()):
            signals += 0.5

        # Short questions without context are low-clarity
        question_only = text.strip().endswith("?") and len(text.split()) < 10
        if question_only:
            signals -= 0.8

        # Reasonable sentence count (1-5 sentences is focused)
        sentence_count = len(re.findall(r'[.!?]+', text))
        if 1 <= sentence_count <= 5:
            signals += 1
        elif sentence_count > 10:
            signals -= 0.5

        # Reasonable length
        word_count = len(text.split())
        if 10 <= word_count <= 300:
            signals += 1
        elif word_count < 10:
            signals += 0.2

        # Markdown formatting
        if "```" in text or "**" in text or "`" in text:
            signals += 0.8

        # Clear scope boundary
        scope_words = r'\b(only|just|specifically|in this file|this function|this class)\b'
        if re.search(scope_words, text, re.IGNORECASE):
            signals += 1

        signals = max(0, signals)
        return sigmoid(signals, 3, 0.6)

    def _score_efficiency(self, turn: ConversationTurn) -> float:
        """Score 0-1: Token economy, optimal prompt length.

        Benchmark: Optimal prompt length is ~40-150 words (bell curve).
        """
        word_count = len(turn.content.split())
        return bell(word_count, 80, 60)

    def _score_chain_of_thought(self, turn: ConversationTurn) -> float:
        """Score 0-1: Structured reasoning in prompts.

        Benchmark: METR finds structured reasoning produces better outcomes.
        Signals: numbered steps, causal language, acceptance criteria,
        expected behavior + constraints in same prompt.
        """
        text = turn.content
        signals = 0

        # Numbered steps or structured lists
        if re.search(r'^\s*\d+[\.\)]\s', text, re.MULTILINE):
            signals += 1

        # Causal/reasoning language
        reasoning = r'\b(because|since|therefore|so that|in order to|this way)\b'
        if re.search(reasoning, text, re.IGNORECASE):
            signals += 1

        # Acceptance criteria or expected behavior
        criteria = r'\b(acceptance criteria|expected|should (return|output|result|produce|be)|must (return|output|be))\b'
        if re.search(criteria, text, re.IGNORECASE):
            signals += 0.8

        # Combined expected behavior + constraints (both in same prompt)
        has_expected = bool(re.search(r'\b(should|expect|want|goal|output)\b', text, re.IGNORECASE))
        has_constraints = bool(re.search(r'\b(without|don\'t|must not|keep|preserve)\b', text, re.IGNORECASE))
        if has_expected and has_constraints:
            signals += 1

        return sigmoid(signals, 2, 0.8)
