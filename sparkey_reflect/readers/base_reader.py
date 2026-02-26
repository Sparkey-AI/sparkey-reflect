"""
Base Reader

Abstract base class for AI tool conversation readers.
Each tool-specific reader implements data extraction from its native storage.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional, Tuple

from sparkey_reflect.core.models import RuleFileInfo, Session, ToolType


class BaseReader(ABC):
    """Abstract reader for AI tool conversation data."""

    @abstractmethod
    def get_tool_type(self) -> ToolType:
        """Return the tool type this reader handles."""

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this tool's data is accessible on the current machine."""

    @abstractmethod
    def get_data_locations(self) -> List[str]:
        """Return paths where this tool stores data."""

    @abstractmethod
    def get_history_range(self) -> Optional[Tuple[datetime, datetime]]:
        """
        Return the earliest and latest available data timestamps.
        Returns None if no data is available.
        """

    @abstractmethod
    def read_sessions(
        self,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        workspace_path: Optional[str] = None,
    ) -> List[Session]:
        """
        Read conversation sessions from the tool's local storage.

        Args:
            since: Only include sessions after this timestamp.
            until: Only include sessions before this timestamp.
            workspace_path: Filter to a specific project/workspace.

        Returns:
            List of Session objects with full turn data.
        """

    @abstractmethod
    def read_rule_files(
        self,
        workspace_path: Optional[str] = None,
    ) -> List[RuleFileInfo]:
        """
        Read and parse rule/instruction files for the tool.

        Args:
            workspace_path: Project root to search. If None, uses cwd.

        Returns:
            List of RuleFileInfo objects describing each config file found.
        """

    def get_status(self) -> dict:
        """Return a summary of tool data availability."""
        available = self.is_available()
        history = self.get_history_range() if available else None
        return {
            "tool": self.get_tool_type().value,
            "available": available,
            "data_locations": self.get_data_locations() if available else [],
            "earliest_data": history[0].isoformat() if history else None,
            "latest_data": history[1].isoformat() if history else None,
        }
