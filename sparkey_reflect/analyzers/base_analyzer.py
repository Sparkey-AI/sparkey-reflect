"""
Base Reflect Analyzer

Abstract interface for all Reflect analyzers.
"""

from abc import ABC, abstractmethod
from typing import List, Optional

from sparkey_reflect.core.models import AnalysisResult, RuleFileInfo, Session


class BaseReflectAnalyzer(ABC):
    """Base class for Reflect analyzers."""

    @abstractmethod
    def get_key(self) -> str:
        """Unique analyzer key."""

    @abstractmethod
    def get_name(self) -> str:
        """Human-readable analyzer name."""

    @abstractmethod
    def analyze(
        self,
        sessions: List[Session],
        rule_files: Optional[List[RuleFileInfo]] = None,
    ) -> AnalysisResult:
        """
        Analyze sessions and optionally rule files.

        Args:
            sessions: List of conversation sessions to analyze.
            rule_files: Optional list of rule files for context.

        Returns:
            AnalysisResult with score (0-100), metrics, and insights.
        """
