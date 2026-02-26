"""
Reflect Analyzer Registry

Follows the AnalyzerRegistry pattern from evaluators/code/config/analyzer_config.py.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from sparkey_reflect.analyzers.base_analyzer import BaseReflectAnalyzer


@dataclass
class ReflectAnalyzerDefinition:
    """Definition of a Reflect analyzer."""
    name: str
    key: str
    description: str
    enabled_by_default: bool = True
    category: str = "core"  # core, tool_specific, advanced
    requires_git: bool = False
    applies_to: List[str] = field(default_factory=lambda: ["claude_code", "cursor", "copilot"])


class ReflectAnalyzerRegistry:
    """Registry of all Reflect analyzers."""

    ANALYZERS = {
        "prompt_quality": ReflectAnalyzerDefinition(
            name="Prompt Quality",
            key="prompt_quality",
            description="Measures prompt specificity, context richness, clarity, and efficiency",
            category="core",
        ),
        "conversation_flow": ReflectAnalyzerDefinition(
            name="Conversation Flow",
            key="conversation_flow",
            description="Analyzes turns-to-resolution, correction rate, and context loss",
            category="core",
        ),
        "context_management": ReflectAnalyzerDefinition(
            name="Context Management",
            key="context_management",
            description="Evaluates file references, error inclusion, and scope clarity",
            category="core",
        ),
        "tool_usage": ReflectAnalyzerDefinition(
            name="Tool Usage",
            key="tool_usage",
            description="Tracks MCP tools, slash commands, and automation opportunities",
            category="tool_specific",
            applies_to=["claude_code", "cursor"],
        ),
        "rule_file": ReflectAnalyzerDefinition(
            name="Rule File Quality",
            key="rule_file",
            description="Analyzes instruction file completeness, specificity, and actionability",
            category="core",
        ),
        "session_patterns": ReflectAnalyzerDefinition(
            name="Session Patterns",
            key="session_patterns",
            description="Detects duration patterns, frequency, task types, and fatigue indicators",
            category="core",
        ),
        "completion_patterns": ReflectAnalyzerDefinition(
            name="Completion Patterns",
            key="completion_patterns",
            description="Copilot-specific acceptance rate and suggestion quality trends",
            category="tool_specific",
            applies_to=["copilot"],
        ),
        "outcome_tracker": ReflectAnalyzerDefinition(
            name="Outcome Tracker",
            key="outcome_tracker",
            description="Correlates AI sessions with git commits and rework rates",
            category="advanced",
            requires_git=True,
        ),
    }

    @classmethod
    def get_all(cls) -> Dict[str, ReflectAnalyzerDefinition]:
        return cls.ANALYZERS

    @classmethod
    def get(cls, key: str) -> Optional[ReflectAnalyzerDefinition]:
        return cls.ANALYZERS.get(key)

    @classmethod
    def get_defaults(cls) -> List[str]:
        return [k for k, a in cls.ANALYZERS.items() if a.enabled_by_default]

    @classmethod
    def get_by_category(cls, category: str) -> Dict[str, ReflectAnalyzerDefinition]:
        return {k: a for k, a in cls.ANALYZERS.items() if a.category == category}

    @classmethod
    def get_for_tool(cls, tool: str) -> List[str]:
        return [k for k, a in cls.ANALYZERS.items() if tool in a.applies_to]


class ReflectAnalyzerConfig:
    """Configuration for which analyzers to run."""

    def __init__(
        self,
        enabled: Optional[List[str]] = None,
        disabled: Optional[List[str]] = None,
        tool: Optional[str] = None,
        skip_git: bool = False,
    ):
        if enabled is not None:
            self.analyzers_to_run = set(enabled)
        else:
            self.analyzers_to_run = set(ReflectAnalyzerRegistry.get_defaults())
            if disabled:
                self.analyzers_to_run -= set(disabled)

        if tool:
            tool_analyzers = set(ReflectAnalyzerRegistry.get_for_tool(tool))
            self.analyzers_to_run &= tool_analyzers

        if skip_git:
            self.analyzers_to_run = {
                k for k in self.analyzers_to_run
                if not (ReflectAnalyzerRegistry.get(k) or ReflectAnalyzerDefinition("", k, "")).requires_git
            }

    def should_run(self, key: str) -> bool:
        return key in self.analyzers_to_run

    def get_enabled(self) -> Set[str]:
        return self.analyzers_to_run


class ReflectAnalyzerPresets:
    """Preset analyzer configurations."""

    @staticmethod
    def quick() -> ReflectAnalyzerConfig:
        """Core analyzers only (fast)."""
        return ReflectAnalyzerConfig(
            enabled=["prompt_quality", "conversation_flow", "session_patterns"]
        )

    @staticmethod
    def coaching() -> ReflectAnalyzerConfig:
        """Standard coaching set (5 analyzers)."""
        return ReflectAnalyzerConfig(
            enabled=[
                "prompt_quality", "conversation_flow",
                "context_management", "session_patterns", "rule_file",
            ]
        )

    @staticmethod
    def full() -> ReflectAnalyzerConfig:
        """All analyzers."""
        return ReflectAnalyzerConfig(
            enabled=list(ReflectAnalyzerRegistry.get_all().keys())
        )

    @staticmethod
    def copilot() -> ReflectAnalyzerConfig:
        """Copilot-focused set."""
        return ReflectAnalyzerConfig(
            enabled=[
                "completion_patterns", "session_patterns",
                "outcome_tracker", "prompt_quality",
            ]
        )
