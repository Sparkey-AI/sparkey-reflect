"""
Session Patterns Analyzer

Analyzes patterns across sessions over time:
- Average session duration and distribution
- Sessions per day / frequency
- Peak usage hours
- Task type distribution
- Fatigue indicators (declining quality in long sessions)
"""

from collections import Counter
from datetime import datetime, timezone
from typing import Dict, List, Optional

from sparkey_reflect.analyzers.base_analyzer import BaseReflectAnalyzer
from sparkey_reflect.core.models import (
    AnalysisResult,
    RuleFileInfo,
    Session,
)


class SessionPatternsAnalyzer(BaseReflectAnalyzer):
    """Analyzes session habits and usage patterns."""

    def get_key(self) -> str:
        return "session_patterns"

    def get_name(self) -> str:
        return "Session Patterns"

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

        durations = [s.duration_minutes for s in sessions if s.duration_minutes > 0]
        avg_duration = sum(durations) / len(durations) if durations else 0

        # Sessions per day
        days = self._get_active_days(sessions)
        sessions_per_day = len(sessions) / max(len(days), 1)

        # Peak hours
        peak_hours = self._compute_peak_hours(sessions)

        # Task type distribution
        type_counts = Counter(s.session_type.value for s in sessions)
        type_dist = {k: v / len(sessions) for k, v in type_counts.items()}

        # Fatigue detection
        fatigue_rate = self._detect_fatigue(sessions)

        # Token efficiency (tokens per minute of session)
        total_tokens = sum(s.total_tokens for s in sessions)
        total_minutes = sum(durations)
        tokens_per_minute = total_tokens / total_minutes if total_minutes > 0 else 0

        # Scoring
        # Good duration: 10-60 min (not too short, not marathon)
        duration_score = self._score_duration(avg_duration)
        # Good frequency: 2-8 sessions/day
        frequency_score = self._score_frequency(sessions_per_day)
        # Diversity: using AI for varied tasks
        diversity_score = self._score_diversity(type_dist)
        # Low fatigue
        fatigue_score = max(0, 25 * (1 - fatigue_rate * 2))

        overall = duration_score + frequency_score + diversity_score + fatigue_score

        period_start = min((s.start_time for s in sessions if s.start_time), default=None)
        period_end = max((s.end_time for s in sessions if s.end_time), default=None)

        return AnalysisResult(
            analyzer_key=self.get_key(),
            analyzer_name=self.get_name(),
            score=max(0, min(100, overall)),
            metrics={
                "avg_duration_minutes": round(avg_duration, 1),
                "sessions_per_day": round(sessions_per_day, 2),
                "total_sessions": len(sessions),
                "active_days": len(days),
                "peak_hour": peak_hours[0] if peak_hours else None,
                "fatigue_rate": round(fatigue_rate, 3),
                "tokens_per_minute": round(tokens_per_minute, 1),
                "task_type_distribution": {k: round(v, 3) for k, v in type_dist.items()},
            },
            insights=[],
            session_count=len(sessions),
            period_start=period_start,
            period_end=period_end,
        )

    # =========================================================================
    # Helper Methods
    # =========================================================================

    def _get_active_days(self, sessions: List[Session]) -> List[str]:
        """Get unique days with sessions."""
        days = set()
        for s in sessions:
            if s.start_time:
                days.add(s.start_time.strftime("%Y-%m-%d"))
        return sorted(days)

    def _compute_peak_hours(self, sessions: List[Session]) -> List[int]:
        """Return top 3 most active hours."""
        hour_counts: Dict[int, int] = Counter()
        for s in sessions:
            if s.start_time:
                hour_counts[s.start_time.hour] += 1
        if not hour_counts:
            return []
        return [h for h, _ in hour_counts.most_common(3)]

    def _detect_fatigue(self, sessions: List[Session]) -> float:
        """
        Detect fatigue: proportion of sessions where quality
        likely degrades in later turns (longer sessions with
        increasing correction patterns).
        """
        fatigue_count = 0
        analyzable = 0

        for session in sessions:
            if session.duration_minutes < 30 or session.user_turn_count < 4:
                continue
            analyzable += 1

            user_turns = [t for t in session.turns if t.role == "user" and t.content]
            if len(user_turns) < 4:
                continue

            # Compare first half vs second half prompt lengths
            mid = len(user_turns) // 2
            first_half_avg = sum(len(t.content.split()) for t in user_turns[:mid]) / mid
            second_half_avg = sum(len(t.content.split()) for t in user_turns[mid:]) / (len(user_turns) - mid)

            # Shorter prompts in second half suggest fatigue (less effort)
            if second_half_avg < first_half_avg * 0.6:
                fatigue_count += 1

        return fatigue_count / analyzable if analyzable > 0 else 0

    def _score_duration(self, avg_minutes: float) -> float:
        """Score 0-25: Ideal session length is 10-60 min."""
        if avg_minutes == 0:
            return 0
        if 10 <= avg_minutes <= 60:
            return 25
        if 5 <= avg_minutes < 10 or 60 < avg_minutes <= 90:
            return 18
        if avg_minutes < 5:
            return 10  # too short, probably not getting value
        return 12  # marathon sessions

    def _score_frequency(self, sessions_per_day: float) -> float:
        """Score 0-25: Ideal frequency is 2-8 sessions/day."""
        if sessions_per_day == 0:
            return 0
        if 2 <= sessions_per_day <= 8:
            return 25
        if 1 <= sessions_per_day < 2 or 8 < sessions_per_day <= 12:
            return 18
        if sessions_per_day < 1:
            return 10  # underutilized
        return 12  # potential over-reliance

    def _score_diversity(self, type_dist: Dict[str, float]) -> float:
        """Score 0-25: Using AI for varied task types is good."""
        active_types = sum(1 for v in type_dist.values() if v > 0.05)
        if active_types >= 4:
            return 25
        if active_types >= 3:
            return 20
        if active_types >= 2:
            return 15
        return 8

