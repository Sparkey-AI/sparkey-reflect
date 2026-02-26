"""
Completion Patterns Analyzer (Copilot-specific)

Evaluates code completion effectiveness across four dimensions:
- Acceptance Rate (0-25): What percentage of suggestions are accepted
- Suggestion Quality (0-25): Are suggestions useful (inferred from acceptance, length, consistency)
- Language Diversity (0-25): Usage across multiple languages/file types
- Latency Performance (0-25): How responsive suggestions are

This analyzer works with both:
- Full conversation sessions (from Reflect trace files)
- Log-derived pseudo-sessions with completion_events in metadata
"""

import logging
from typing import Dict, List, Optional

from sparkey_reflect.analyzers.base_analyzer import BaseReflectAnalyzer
from sparkey_reflect.core.models import (
    AnalysisResult,
    RuleFileInfo,
    Session,
)

logger = logging.getLogger(__name__)


class CompletionPatternsAnalyzer(BaseReflectAnalyzer):
    """Analyzes Copilot completion acceptance rates and suggestion quality."""

    def get_key(self) -> str:
        return "completion_patterns"

    def get_name(self) -> str:
        return "Completion Patterns"

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

        # Extract completion events from session metadata (log-derived sessions)
        all_events = []
        for session in sessions:
            events = session.metadata.get("events", [])
            all_events.extend(events)

        # Score each dimension
        acceptance_score = self._score_acceptance_rate(sessions, all_events)
        quality_score = self._score_suggestion_quality(sessions, all_events)
        diversity_score = self._score_language_diversity(sessions, all_events)
        latency_score = self._score_latency(all_events)

        overall = acceptance_score + quality_score + diversity_score + latency_score

        # Compute aggregate metrics
        total_events = len(all_events)
        accepted_events = sum(1 for e in all_events if e.get("accepted", False))
        acceptance_rate = (accepted_events / total_events * 100) if total_events > 0 else 0
        languages = list(set(
            e.get("language", "unknown") for e in all_events
            if e.get("language") and e.get("language") != "unknown"
        ))

        period_start = min((s.start_time for s in sessions if s.start_time), default=None)
        period_end = max((s.end_time for s in sessions if s.end_time), default=None)

        return AnalysisResult(
            analyzer_key=self.get_key(),
            analyzer_name=self.get_name(),
            score=overall,
            metrics={
                "acceptance_rate": acceptance_rate,
                "total_completions": total_events,
                "accepted_completions": accepted_events,
                "language_count": len(languages),
                "acceptance_score": acceptance_score,
                "quality_score": quality_score,
                "diversity_score": diversity_score,
                "latency_score": latency_score,
            },
            insights=[],
            session_count=len(sessions),
            period_start=period_start,
            period_end=period_end,
            metadata={
                "languages": languages,
                "source_types": list(set(
                    s.metadata.get("source", "unknown") for s in sessions
                )),
            },
        )

    # =========================================================================
    # Scoring Dimensions
    # =========================================================================

    def _score_acceptance_rate(
        self, sessions: List[Session], events: List[Dict],
    ) -> float:
        """Score 0-25: What percentage of suggestions are accepted."""
        if not events:
            # Fall back to session-level metadata
            rates = []
            for s in sessions:
                rate = s.metadata.get("acceptance_rate")
                if rate is not None:
                    rates.append(rate)
            if not rates:
                return 12.5  # neutral score when no data
            avg_rate = sum(rates) / len(rates) * 100
        else:
            total = len(events)
            accepted = sum(1 for e in events if e.get("accepted", False))
            avg_rate = accepted / total * 100 if total > 0 else 0

        # Score tiers:
        # 80%+ = 25, 60-80% = 20, 40-60% = 15, 20-40% = 10, <20% = 5
        if avg_rate >= 80:
            return 25.0
        elif avg_rate >= 60:
            return 20.0 + (avg_rate - 60) / 20 * 5
        elif avg_rate >= 40:
            return 15.0 + (avg_rate - 40) / 20 * 5
        elif avg_rate >= 20:
            return 10.0 + (avg_rate - 20) / 20 * 5
        else:
            return max(2, avg_rate / 20 * 10)

    def _score_suggestion_quality(
        self, sessions: List[Session], events: List[Dict],
    ) -> float:
        """Score 0-25: Inferred suggestion quality from patterns."""
        if not events and not sessions:
            return 12.5

        score = 12.5  # start neutral

        if events:
            # Quality signal 1: Accepted suggestions have reasonable length
            accepted = [e for e in events if e.get("accepted", False)]
            if accepted:
                avg_length = sum(e.get("suggestion_length", 0) for e in accepted) / len(accepted)
                # Sweet spot: 2-10 lines per suggestion
                if 2 <= avg_length <= 10:
                    score += 5
                elif 1 <= avg_length <= 20:
                    score += 3
                elif avg_length > 0:
                    score += 1

            # Quality signal 2: Consistent acceptance (not wildly varying)
            if len(events) >= 10:
                # Check if acceptance is consistent across the period
                half = len(events) // 2
                first_half = events[:half]
                second_half = events[half:]

                first_rate = sum(1 for e in first_half if e.get("accepted")) / len(first_half)
                second_rate = sum(1 for e in second_half if e.get("accepted")) / len(second_half)

                consistency = 1.0 - abs(first_rate - second_rate)
                score += consistency * 5

            # Quality signal 3: Volume (more completions = more engagement)
            events_per_session = len(events) / max(len(sessions), 1)
            if events_per_session >= 20:
                score += 5
            elif events_per_session >= 10:
                score += 3
            elif events_per_session >= 5:
                score += 2
        else:
            # Infer from conversation-based sessions
            for session in sessions:
                if session.metadata.get("acceptance_rate", 0) > 0.5:
                    score += 2
                    break

        return max(0, min(25, score))

    def _score_language_diversity(
        self, sessions: List[Session], events: List[Dict],
    ) -> float:
        """Score 0-25: Usage across multiple languages/file types."""
        languages = set()

        # From events
        for e in events:
            lang = e.get("language", "")
            if lang and lang != "unknown":
                languages.add(lang)

        # From session metadata
        for s in sessions:
            for lang in s.metadata.get("languages", []):
                if lang and lang != "unknown":
                    languages.add(lang)

        count = len(languages)

        # Score tiers:
        # 5+ languages = 25, 4 = 20, 3 = 15, 2 = 10, 1 = 7, 0 = 3
        if count >= 5:
            return 25.0
        elif count == 4:
            return 20.0
        elif count == 3:
            return 15.0
        elif count == 2:
            return 10.0
        elif count == 1:
            return 7.0
        return 3.0

    def _score_latency(self, events: List[Dict]) -> float:
        """Score 0-25: How responsive suggestions are."""
        if not events:
            return 12.5  # neutral when no latency data

        latencies = []
        for e in events:
            lat = e.get("latency_ms")
            if lat is not None and lat > 0:
                latencies.append(lat)

        if not latencies:
            return 12.5

        avg_latency = sum(latencies) / len(latencies)

        # Score tiers:
        # <100ms = 25, <300ms = 22, <500ms = 18, <1000ms = 14, <2000ms = 10, >2000ms = 5
        if avg_latency < 100:
            return 25.0
        elif avg_latency < 300:
            return 22.0
        elif avg_latency < 500:
            return 18.0
        elif avg_latency < 1000:
            return 14.0
        elif avg_latency < 2000:
            return 10.0
        return 5.0

