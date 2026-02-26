"""
Outcome Tracker Analyzer

Correlates AI coding sessions with development outcomes across five dimensions
using smooth scoring curves grounded in industry benchmarks:
- AI Commit Rate (w=0.20): Proportion of commits during/after AI sessions
- Productivity (w=0.20): Commits per session hour (DORA: throughput proxy)
- Rework Rate (w=0.25): Lower rework = better (GitClear 2024: AI rework 3.1->5.7%)
- Quality Signals (w=0.15): Commit message quality and cadence
- Commit Quality Trend (w=0.20): Improving rework rate over time (NEW)

Benchmarks: DORA (throughput proxy), GitClear 2024 (AI rework trends),
DORA (stability improving over time).

Requires git access (workspace_path must be a git repo).
"""

import logging
import re
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
from sparkey_reflect.core.scoring import sigmoid, weighted_sum

logger = logging.getLogger(__name__)

# How close a commit must be to a session to be considered "AI-assisted"
AI_ASSISTED_WINDOW_MINUTES = 30

# Rework patterns in commit messages
REWORK_PATTERNS = [
    r"\b(fix|revert|undo|rollback|hotfix|patch)\b",
    r"\b(typo|oops|again|retry|re-do)\b",
    r"\b(bug|broken|wrong|incorrect)\b",
]


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

        # Rework rate
        rework_rate = self._compute_rework_rate(all_commits)

        # Quality signals (compound score)
        quality_dim = self._compute_quality_dim(all_commits)

        # Commit quality trend: compare recent 25% vs older 75%
        trend_improvement = self._compute_quality_trend(all_commits)

        # Smooth scoring: each dimension 0-1
        commit_rate_dim = sigmoid(ai_commit_rate, 0.4, 5)
        productivity_dim = sigmoid(commits_per_hour, 1.0, 2)
        rework_dim = 1 - sigmoid(rework_rate, 0.12, 10)
        trend_dim = sigmoid(trend_improvement, 0, 3)

        overall = weighted_sum([
            (commit_rate_dim, 0.20),
            (productivity_dim, 0.20),
            (rework_dim, 0.25),
            (quality_dim, 0.15),
            (trend_dim, 0.20),
        ])

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
                "quality_signals": round(quality_dim, 3),
                "commit_quality_trend": round(trend_improvement, 3),
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
                start = session.start_time - window
                end = session.end_time + window
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
        rework_count = 0
        for commit in commits:
            subject = commit["subject"].lower()
            if any(re.search(p, subject) for p in REWORK_PATTERNS):
                rework_count += 1
        return rework_count / len(commits) if commits else 0

    def _compute_quality_dim(self, commits: List[Dict]) -> float:
        """Compute quality dimension (0-1) from commit patterns."""
        if not commits:
            return 0.5

        score = 0.0

        # Good commit message length
        avg_subject_len = sum(len(c["subject"]) for c in commits) / len(commits)
        if avg_subject_len >= 30:
            score += 0.35
        elif avg_subject_len >= 15:
            score += 0.2

        # Consistent cadence
        if len(commits) >= 3:
            timestamps = sorted(c["timestamp"] for c in commits)
            gaps = [
                (timestamps[i + 1] - timestamps[i]).total_seconds() / 3600
                for i in range(len(timestamps) - 1)
            ]
            avg_gap = sum(gaps) / len(gaps) if gaps else 0
            if 1 <= avg_gap <= 8:
                score += 0.35
            elif 0.5 <= avg_gap <= 24:
                score += 0.2

        # Low-quality message rate
        low_quality_msgs = sum(
            1 for c in commits
            if len(c["subject"]) < 10
            or c["subject"].lower() in ("wip", "update", "fix", "changes", "stuff")
        )
        low_quality_rate = low_quality_msgs / len(commits)
        if low_quality_rate == 0:
            score += 0.3
        elif low_quality_rate < 0.1:
            score += 0.2

        return min(1.0, score)

    def _compute_quality_trend(self, commits: List[Dict]) -> float:
        """Compare rework rate in recent 25% vs older 75%.

        Returns positive value if improving (recent has less rework),
        negative if declining, ~0 if stable. Range roughly -1 to 1.
        """
        if len(commits) < 4:
            return 0.0  # insufficient data -> neutral

        # Sort by timestamp (oldest first)
        sorted_commits = sorted(commits, key=lambda c: c["timestamp"])
        split_idx = int(len(sorted_commits) * 0.75)

        older = sorted_commits[:split_idx]
        recent = sorted_commits[split_idx:]

        def rework_rate(commit_list: List[Dict]) -> float:
            count = sum(
                1 for c in commit_list
                if any(re.search(p, c["subject"].lower()) for p in REWORK_PATTERNS)
            )
            return count / len(commit_list) if commit_list else 0

        older_rate = rework_rate(older)
        recent_rate = rework_rate(recent)

        # Positive = improving (recent rework is lower)
        return older_rate - recent_rate
