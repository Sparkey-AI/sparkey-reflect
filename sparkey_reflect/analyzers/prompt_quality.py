"""
Prompt Quality Analyzer

Evaluates the quality of user prompts across four dimensions:
- Specificity (0-25): How precise and targeted the prompt is
- Context Richness (0-25): File refs, error context, code snippets provided
- Clarity (0-25): Structure, actionable language, unambiguous intent
- Efficiency (0-25): Token economy, avoiding redundancy, right-sizing requests
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

        for session in sessions:
            user_turns = [t for t in session.turns if t.role == "user" and t.content]
            for turn in user_turns:
                all_specificity.append(self._score_specificity(turn))
                all_context.append(self._score_context_richness(turn))
                all_clarity.append(self._score_clarity(turn))
                all_efficiency.append(self._score_efficiency(turn, session))

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
        overall = specificity + context + clarity + efficiency

        period_start = min((s.start_time for s in sessions if s.start_time), default=None)
        period_end = max((s.end_time for s in sessions if s.end_time), default=None)

        return AnalysisResult(
            analyzer_key=self.get_key(),
            analyzer_name=self.get_name(),
            score=overall,
            metrics={
                "specificity": specificity,
                "context_richness": context,
                "clarity": clarity,
                "efficiency": efficiency,
                "prompts_analyzed": len(all_specificity),
            },
            insights=[],
            session_count=len(sessions),
            period_start=period_start,
            period_end=period_end,
        )

    # =========================================================================
    # Scoring Dimensions
    # =========================================================================

    def _score_specificity(self, turn: ConversationTurn) -> float:
        """Score 0-25: How specific and targeted the prompt is."""
        text = turn.content
        score = 0.0
        word_count = len(text.split())

        # Penalize very short prompts (< 5 words)
        if word_count < 5:
            return min(score + 3, 25)

        # Mentions specific file names or paths
        if turn.file_references:
            score += 5

        # Contains function/class/variable names (camelCase, snake_case, PascalCase)
        identifiers = re.findall(r'\b[a-z_][a-zA-Z0-9_]{2,}\b', text)
        if len(identifiers) >= 2:
            score += 4
        elif identifiers:
            score += 2

        # Contains line numbers or specific locations
        if re.search(r'line\s+\d+|:\d+', text):
            score += 3

        # Specific action verbs (not vague)
        specific_verbs = r'\b(add|remove|rename|change|fix|update|create|delete|move|extract|refactor|implement|replace|modify)\b'
        if re.search(specific_verbs, text, re.IGNORECASE):
            score += 4

        # Mentions specific technologies/frameworks
        tech_pattern = r'\b(React|Python|TypeScript|FastAPI|SQLAlchemy|Postgres|Redis|Docker|AWS|jest|pytest)\b'
        if re.search(tech_pattern, text, re.IGNORECASE):
            score += 3

        # Penalize vague prompts
        vague_patterns = r'\b(help me|can you|please|something|anything|stuff|things|somehow)\b'
        vague_count = len(re.findall(vague_patterns, text, re.IGNORECASE))
        score -= vague_count * 1.5

        # Well-scoped length (20-200 words is ideal)
        if 20 <= word_count <= 200:
            score += 4
        elif word_count > 200:
            score += 2

        return max(0, min(25, score))

    def _score_context_richness(self, turn: ConversationTurn) -> float:
        """Score 0-25: How much useful context the prompt includes."""
        score = 0.0

        # File references
        ref_count = len(turn.file_references)
        if ref_count >= 3:
            score += 7
        elif ref_count >= 1:
            score += 4

        # Error context included
        if turn.has_error_context:
            score += 6

        # Code snippets included
        if turn.has_code_snippet:
            score += 5

        # Expected behavior described
        expected = r'\b(should|expect|want|need|goal|output|result|return)\b'
        if re.search(expected, turn.content, re.IGNORECASE):
            score += 4

        # Constraints mentioned
        constraints = r'\b(without|don\'t|must not|keep|preserve|maintain|backward.?compat)\b'
        if re.search(constraints, turn.content, re.IGNORECASE):
            score += 3

        return max(0, min(25, score))

    def _score_clarity(self, turn: ConversationTurn) -> float:
        """Score 0-25: Structural clarity and unambiguous intent."""
        text = turn.content
        score = 0.0

        # Structured with numbered steps or bullet points
        if re.search(r'^\s*(\d+[\.\)]\s|[-*]\s)', text, re.MULTILINE):
            score += 5

        # Has a clear imperative sentence
        if re.match(r'^[A-Z][a-z]', text.strip()):
            score += 2

        # Not a question-only prompt (questions are fine but should have context)
        question_only = text.strip().endswith("?") and len(text.split()) < 10
        if question_only:
            score -= 3

        # Single clear task (not multiple unrelated requests)
        sentence_count = len(re.findall(r'[.!?]+', text))
        if 1 <= sentence_count <= 5:
            score += 4
        elif sentence_count > 10:
            score -= 2  # too many things at once

        # Reasonable length (not too short, not a wall of text)
        word_count = len(text.split())
        if 10 <= word_count <= 300:
            score += 5
        elif word_count < 10:
            score += 1

        # Uses markdown formatting
        if "```" in text or "**" in text or "`" in text:
            score += 3

        # Clear scope boundary
        scope_words = r'\b(only|just|specifically|in this file|this function|this class)\b'
        if re.search(scope_words, text, re.IGNORECASE):
            score += 4

        return max(0, min(25, score))

    def _score_efficiency(self, turn: ConversationTurn, session: Session) -> float:
        """Score 0-25: Token economy and avoiding redundancy."""
        text = turn.content
        word_count = len(text.split())
        score = 15.0  # start with decent score, penalize inefficiency

        # Very long prompts (>500 words) suggest inefficiency
        if word_count > 500:
            score -= 5
        elif word_count > 300:
            score -= 2

        # Repetitive content (same phrases repeated)
        sentences = re.split(r'[.!?\n]+', text)
        unique_sentences = set(s.strip().lower() for s in sentences if s.strip())
        if len(sentences) > 3 and len(unique_sentences) < len(sentences) * 0.7:
            score -= 4

        # Excessive pleasantries / filler
        fillers = r'\b(please|thank you|thanks|could you|would you|I was wondering|I think maybe)\b'
        filler_count = len(re.findall(fillers, text, re.IGNORECASE))
        score -= filler_count * 0.5

        # Contains unnecessary preamble
        preamble = r'^(hi|hello|hey|so|ok|okay|well|alright)\b'
        if re.match(preamble, text.strip(), re.IGNORECASE):
            score -= 1

        # Good info-to-token ratio: short prompts with file refs and code = efficient
        if word_count < 100 and (turn.file_references or turn.has_code_snippet):
            score += 5

        return max(0, min(25, score))

