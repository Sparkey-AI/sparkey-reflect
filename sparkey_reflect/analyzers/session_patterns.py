"""
Session Patterns Analyzer

Analyzes patterns across sessions over time using smooth scoring curves
grounded in industry benchmarks:
- Duration (w=0.20): Ideal 25-45 min = peak productivity (DevEx)
- Frequency (w=0.20): 2-6/day = sustained engagement (SPACE)
- Diversity (w=0.15): Using AI for varied task types
- Fatigue (w=0.20): Quality degrades after threshold
- Deep Work Alignment (w=0.25): Uninterrupted coding blocks (DORA 2024) (NEW)

Benchmarks: DevEx (25-45 min sessions = peak productivity), SPACE (2-6/day
sustained engagement), DORA 2024 (uninterrupted coding blocks boost throughput).
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
from sparkey_reflect.core.scoring import bell, diminishing, sigmoid, weighted_sum

# Minimum gap (minutes) between session end and next session start
# to consider the block "uninterrupted"
DEEP_WORK_GAP_MINUTES = 15

# Minimum block duration (minutes) to count as deep work
DEEP_WORK_MIN_BLOCK_MINUTES = 120


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
        active_types = sum(1 for v in type_dist.values() if v > 0.05)

        # Fatigue detection
        fatigue_rate = self._detect_fatigue(sessions)

        # Token efficiency
        total_tokens = sum(s.total_tokens for s in sessions)
        total_minutes = sum(durations)
        tokens_per_minute = total_tokens / total_minutes if total_minutes > 0 else 0

        # Deep work alignment
        deep_work_ratio = self._compute_deep_work_ratio(sessions)

        # Smooth scoring: each dimension 0-1
        duration_dim = bell(avg_duration, 35, 20)
        frequency_dim = bell(sessions_per_day, 4, 2.5)
        diversity_dim = diminishing(active_types, 5)
        fatigue_dim = 1 - sigmoid(fatigue_rate, 0.2, 8)
        deep_work_dim = sigmoid(deep_work_ratio, 0.4, 4)

        overall = weighted_sum([
            (duration_dim, 0.20),
            (frequency_dim, 0.20),
            (diversity_dim, 0.15),
            (fatigue_dim, 0.20),
            (deep_work_dim, 0.25),
        ])

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
                "deep_work_ratio": round(deep_work_ratio, 3),
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

            mid = len(user_turns) // 2
            first_half_avg = sum(len(t.content.split()) for t in user_turns[:mid]) / mid
            second_half_avg = sum(len(t.content.split()) for t in user_turns[mid:]) / (len(user_turns) - mid)

            if second_half_avg < first_half_avg * 0.6:
                fatigue_count += 1

        return fatigue_count / analyzable if analyzable > 0 else 0

    def _compute_deep_work_ratio(self, sessions: List[Session]) -> float:
        """Fraction of sessions occurring in 2+ hour uninterrupted blocks.

        Based on DORA 2024 finding that fragmented work hurts throughput.
        A block is uninterrupted if no session starts within 15 min of
        the previous session's end.
        """
        timed_sessions = [s for s in sessions if s.start_time and s.end_time]
        if not timed_sessions:
            return 0.0

        timed_sessions.sort(key=lambda s: s.start_time)

        # Group sessions into contiguous blocks
        blocks: List[List[Session]] = []
        current_block: List[Session] = [timed_sessions[0]]

        for s in timed_sessions[1:]:
            prev = current_block[-1]
            gap_minutes = (s.start_time - prev.end_time).total_seconds() / 60
            if gap_minutes <= DEEP_WORK_GAP_MINUTES:
                current_block.append(s)
            else:
                blocks.append(current_block)
                current_block = [s]
        blocks.append(current_block)

        # Count sessions in blocks that span 2+ hours
        deep_work_sessions = 0
        for block in blocks:
            block_start = block[0].start_time
            block_end = block[-1].end_time
            block_duration = (block_end - block_start).total_seconds() / 60
            if block_duration >= DEEP_WORK_MIN_BLOCK_MINUTES:
                deep_work_sessions += len(block)

        return deep_work_sessions / len(timed_sessions)
