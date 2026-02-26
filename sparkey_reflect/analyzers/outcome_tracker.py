"""
Outcome Tracker Analyzer

Correlates AI coding sessions with development outcomes:
- AI-Assisted Commit Rate (0-25): What proportion of commits happen during/after AI sessions
- Session Productivity (0-25): Tokens and turns per productive output
- Rework Indicator (0-25): Whether AI sessions correlate with lower rework
- Quality Signals (0-25): Session patterns associated with higher-quality outcomes

Requires git access (workspace_path must be a git repo).
"""

import logging
import subprocess
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

from sparkey_reflect.analyzers.base_analyzer import BaseReflectAnalyzer
from sparkey_reflect.core.models import (
    AnalysisResult,
    InsightCategory,
    InsightSeverity,
    ReflectInsight,
    RuleFileInfo,
    Session,
)

# Note: InsightCategory, InsightSeverity, ReflectInsight still needed
# for the early-return "No git data available" insight in analyze().

logger = logging.getLogger(__name__)

# How close a commit must be to a session to be considered "AI-assisted"
AI_ASSISTED_WINDOW_MINUTES = 30


class OutcomeTrackerAnalyzer(BaseReflectAnalyzer):
    """Correlates AI sessions with git outcomes."""

    def get_key(self) -> str:
        return "outcome_tracker"

    def get_name(self) -> str:
        return "Outcome Tracker"

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

        # Collect unique workspace paths
        workspaces = set()
        for s in sessions:
            if s.workspace_path:
                workspaces.add(s.workspace_path)

        # Get git commits for each workspace
        all_commits: List[Dict] = []
        for ws in workspaces:
            period_start = min(
                (s.start_time for s in sessions if s.start_time),
                default=datetime.now(timezone.utc) - timedelta(days=30),
            )
            commits = self._get_git_commits(ws, since=period_start)
            all_commits.extend(commits)

        if not all_commits:
            return AnalysisResult(
                analyzer_key=self.get_key(),
                analyzer_name=self.get_name(),
                score=50,  # neutral if no git data
                metrics={
                    "git_available": False,
                    "commits_found": 0,
                },
                insights=[ReflectInsight(
                    category=InsightCategory.OUTCOME_QUALITY,
                    title="No git data available",
                    severity=InsightSeverity.INFO,
                    recommendation=(
                        "Outcome tracking requires git history. Ensure your "
                        "workspace path points to a git repository."
                    ),
                    evidence="No commits found in workspace paths",
                    metric_key="commits_found",
                    metric_value=0,
                )],
                session_count=len(sessions),
            )

        # Correlate commits with sessions
        ai_assisted, total_commits = self._correlate_commits(sessions, all_commits)
        ai_commit_rate = ai_assisted / total_commits if total_commits > 0 else 0

        # Session productivity: commits per session hour
        total_session_hours = sum(s.duration_minutes for s in sessions) / 60.0
        commits_per_hour = total_commits / total_session_hours if total_session_hours > 0 else 0

        # Rework indicator: analyze commit message patterns for rework signals
        rework_rate = self._compute_rework_rate(all_commits)

        # Quality signals: multi-file commits, reasonable sizes
        quality_score = self._compute_quality_signals(all_commits)

        # Scoring
        commit_rate_score = self._score_ai_commit_rate(ai_commit_rate)
        productivity_score = self._score_productivity(commits_per_hour, total_session_hours)
        rework_score = self._score_rework(rework_rate)
        quality_dim_score = quality_score  # already 0-25

        overall = commit_rate_score + productivity_score + rework_score + quality_dim_score

        period_start = min((s.start_time for s in sessions if s.start_time), default=None)
        period_end = max((s.end_time for s in sessions if s.end_time), default=None)

        return AnalysisResult(
            analyzer_key=self.get_key(),
            analyzer_name=self.get_name(),
            score=max(0, min(100, overall)),
            metrics={
                "git_available": True,
                "total_commits": total_commits,
                "ai_assisted_commits": ai_assisted,
                "ai_commit_rate": round(ai_commit_rate, 3),
                "commits_per_hour": round(commits_per_hour, 2),
                "rework_rate": round(rework_rate, 3),
                "quality_score": round(quality_score, 1),
                "total_session_hours": round(total_session_hours, 1),
            },
            insights=[],
            session_count=len(sessions),
            period_start=period_start,
            period_end=period_end,
        )

    # =========================================================================
    # Git Integration
    # =========================================================================

    def _get_git_commits(
        self, workspace: str, since: Optional[datetime] = None
    ) -> List[Dict]:
        """Get git commits from a workspace."""
        try:
            cmd = [
                "git", "-C", workspace, "log",
                "--format=%H|%aI|%s|%an",
                "--no-merges",
            ]
            if since:
                cmd.append(f"--since={since.isoformat()}")

            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=15
            )
            if result.returncode != 0:
                return []

            commits = []
            for line in result.stdout.strip().split("\n"):
                if not line:
                    continue
                parts = line.split("|", 3)
                if len(parts) < 4:
                    continue
                sha, date_str, subject, author = parts
                try:
                    ts = datetime.fromisoformat(date_str)
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=timezone.utc)
                except ValueError:
                    continue
                commits.append({
                    "sha": sha,
                    "timestamp": ts,
                    "subject": subject,
                    "author": author,
                    "workspace": workspace,
                })
            return commits

        except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
            logger.debug("Git error for %s: %s", workspace, e)
            return []

    def _correlate_commits(
        self, sessions: List[Session], commits: List[Dict]
    ) -> Tuple[int, int]:
        """Count how many commits fall within AI session windows."""
        ai_assisted = 0
        window = timedelta(minutes=AI_ASSISTED_WINDOW_MINUTES)

        for commit in commits:
            ct = commit["timestamp"]
            for session in sessions:
                if not session.start_time or not session.end_time:
                    continue
                # Commit within session timeframe (± window)
                start = session.start_time - window
                end = session.end_time + window
                # Ensure timezone-aware comparison
                if ct.tzinfo is None:
                    ct = ct.replace(tzinfo=timezone.utc)
                if start.tzinfo is None:
                    start = start.replace(tzinfo=timezone.utc)
                if end.tzinfo is None:
                    end = end.replace(tzinfo=timezone.utc)
                if start <= ct <= end:
                    ai_assisted += 1
                    break

        return ai_assisted, len(commits)

    def _compute_rework_rate(self, commits: List[Dict]) -> float:
        """Estimate rework rate from commit messages."""
        rework_patterns = [
            r"\b(fix|revert|undo|rollback|hotfix|patch)\b",
            r"\b(typo|oops|again|retry|re-do)\b",
            r"\b(bug|broken|wrong|incorrect)\b",
        ]
        rework_count = 0
        for commit in commits:
            subject = commit["subject"].lower()
            if any(
                __import__("re").search(p, subject)
                for p in rework_patterns
            ):
                rework_count += 1
        return rework_count / len(commits) if commits else 0

    def _compute_quality_signals(self, commits: List[Dict]) -> float:
        """Score 0-25: Quality signals from commit patterns."""
        if not commits:
            return 12

        score = 12.0  # baseline

        # Good commit message length (not too short)
        avg_subject_len = sum(len(c["subject"]) for c in commits) / len(commits)
        if avg_subject_len >= 30:
            score += 5
        elif avg_subject_len >= 15:
            score += 3

        # Consistent committing (not all at once)
        if len(commits) >= 3:
            timestamps = sorted(c["timestamp"] for c in commits)
            gaps = [
                (timestamps[i + 1] - timestamps[i]).total_seconds() / 3600
                for i in range(len(timestamps) - 1)
            ]
            avg_gap = sum(gaps) / len(gaps) if gaps else 0
            # Regular commits (gap of 1-8 hours) = good cadence
            if 1 <= avg_gap <= 8:
                score += 5
            elif 0.5 <= avg_gap <= 24:
                score += 3

        # Descriptive commit messages (not just "wip" or "update")
        low_quality_msgs = sum(
            1 for c in commits
            if len(c["subject"]) < 10
            or c["subject"].lower() in ("wip", "update", "fix", "changes", "stuff")
        )
        if low_quality_msgs == 0:
            score += 3
        elif low_quality_msgs / len(commits) < 0.1:
            score += 2

        return min(25, score)

    # =========================================================================
    # Scoring Dimensions (0-25 each)
    # =========================================================================

    def _score_ai_commit_rate(self, rate: float) -> float:
        """Score 0-25: AI-assisted commit rate."""
        # Higher rate means AI sessions correlate with productive output
        if rate >= 0.7:
            return 25
        if rate >= 0.5:
            return 21
        if rate >= 0.3:
            return 16
        if rate >= 0.1:
            return 10
        return 5

    def _score_productivity(self, commits_per_hour: float, total_hours: float) -> float:
        """Score 0-25: Commits per session hour."""
        if total_hours < 0.5:
            return 12  # insufficient data

        if commits_per_hour >= 2:
            return 25
        if commits_per_hour >= 1:
            return 21
        if commits_per_hour >= 0.5:
            return 16
        if commits_per_hour >= 0.2:
            return 10
        return 5

    def _score_rework(self, rework_rate: float) -> float:
        """Score 0-25: Lower rework = better."""
        if rework_rate <= 0.05:
            return 25
        if rework_rate <= 0.1:
            return 21
        if rework_rate <= 0.2:
            return 16
        if rework_rate <= 0.35:
            return 10
        return 5

