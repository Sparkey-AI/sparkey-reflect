"""
Sparkey Reflect Data Models

Core dataclasses and enums for the Reflect analysis engine.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any


# =============================================================================
# Enums
# =============================================================================

class ToolType(str, Enum):
    """Supported AI coding tools."""
    CLAUDE_CODE = "claude_code"
    CURSOR = "cursor"
    COPILOT = "copilot"


class SessionType(str, Enum):
    """Classification of coding session purpose."""
    CODING = "coding"
    DEBUGGING = "debugging"
    REFACTORING = "refactoring"
    DOCS = "docs"
    TESTING = "testing"
    EXPLORATION = "exploration"
    UNKNOWN = "unknown"


class InsightSeverity(str, Enum):
    """Severity levels for coaching insights."""
    INFO = "info"
    SUGGESTION = "suggestion"
    WARNING = "warning"
    CRITICAL = "critical"


class InsightCategory(str, Enum):
    """Categories of coaching insights."""
    PROMPT_ENGINEERING = "prompt_engineering"
    CONVERSATION_FLOW = "conversation_flow"
    CONTEXT_MANAGEMENT = "context_management"
    TOOL_MASTERY = "tool_mastery"
    RULE_FILE_QUALITY = "rule_file_quality"
    SESSION_HABITS = "session_habits"
    OUTCOME_QUALITY = "outcome_quality"
    COMPLETION_USAGE = "completion_usage"


class TrendDirection(str, Enum):
    """Direction of a metric trend."""
    IMPROVING = "improving"
    DECLINING = "declining"
    STABLE = "stable"
    INSUFFICIENT_DATA = "insufficient_data"


# =============================================================================
# Core Data Models
# =============================================================================

@dataclass
class ConversationTurn:
    """A single turn in an AI conversation."""
    role: str  # "user", "assistant", "system", "tool_use", "tool_result"
    content: str
    timestamp: Optional[datetime] = None
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    tool_name: Optional[str] = None  # for tool_use/tool_result turns
    input_tokens: int = 0
    output_tokens: int = 0
    file_references: List[str] = field(default_factory=list)
    has_error_context: bool = False
    has_code_snippet: bool = False


@dataclass
class Session:
    """A complete AI coding session."""
    session_id: str
    tool: ToolType
    turns: List[ConversationTurn] = field(default_factory=list)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_minutes: float = 0.0
    workspace_path: Optional[str] = None
    branch: Optional[str] = None
    model: Optional[str] = None
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    session_type: SessionType = SessionType.UNKNOWN
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    @property
    def turn_count(self) -> int:
        return len(self.turns)

    @property
    def user_turn_count(self) -> int:
        return sum(1 for t in self.turns if t.role == "user")

    @property
    def assistant_turn_count(self) -> int:
        return sum(1 for t in self.turns if t.role == "assistant")

    @property
    def tool_use_count(self) -> int:
        return sum(len(t.tool_calls) for t in self.turns)


@dataclass
class CompletionEvent:
    """A code completion event (Copilot-specific)."""
    event_id: str
    timestamp: datetime
    language: str
    suggestion_length: int
    accepted: bool
    latency_ms: Optional[float] = None
    model: Optional[str] = None
    file_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RuleFileInfo:
    """Parsed rule/instruction file metadata."""
    file_path: str
    file_type: str  # "claude_md", "cursorrules", "cursor_mdc", "copilot_instructions", etc.
    tool: ToolType
    exists: bool = True
    word_count: int = 0
    section_count: int = 0
    sections: List[str] = field(default_factory=list)
    has_examples: bool = False
    has_constraints: bool = False
    has_project_context: bool = False
    has_style_guide: bool = False
    has_tool_config: bool = False
    last_modified: Optional[datetime] = None
    raw_content: Optional[str] = None  # only held in memory during analysis, never persisted


# =============================================================================
# Analysis Output Models
# =============================================================================

@dataclass
class AnalysisResult:
    """Output from a single analyzer for a set of sessions."""
    analyzer_key: str
    analyzer_name: str
    score: float  # 0-100 composite score
    metrics: Dict[str, float] = field(default_factory=dict)
    insights: List['ReflectInsight'] = field(default_factory=list)
    session_count: int = 0
    period_start: Optional[datetime] = None
    period_end: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReflectInsight:
    """A coaching insight with actionable recommendation."""
    category: InsightCategory
    title: str
    severity: InsightSeverity
    recommendation: str
    evidence: str  # brief description of what triggered this insight
    metric_key: Optional[str] = None  # which metric triggered it
    metric_value: Optional[float] = None
    trend: TrendDirection = TrendDirection.INSUFFICIENT_DATA
    created_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category.value,
            "title": self.title,
            "severity": self.severity.value,
            "recommendation": self.recommendation,
            "evidence": self.evidence,
            "metric_key": self.metric_key,
            "metric_value": self.metric_value,
            "trend": self.trend.value,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


@dataclass
class ReflectReport:
    """Complete analysis report for a time period."""
    tool: ToolType
    period_start: datetime
    period_end: datetime
    overall_score: float  # 0-100 weighted average
    results: List[AnalysisResult] = field(default_factory=list)
    insights: List[ReflectInsight] = field(default_factory=list)
    session_count: int = 0
    total_turns: int = 0
    total_tokens: int = 0
    total_duration_minutes: float = 0.0
    trends: Dict[str, TrendDirection] = field(default_factory=dict)
    overall_assessment: Optional[str] = None
    created_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool": self.tool.value,
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "overall_score": round(self.overall_score, 1),
            "overall_assessment": self.overall_assessment,
            "session_count": self.session_count,
            "total_turns": self.total_turns,
            "total_tokens": self.total_tokens,
            "total_duration_minutes": round(self.total_duration_minutes, 1),
            "results": [
                {
                    "analyzer": r.analyzer_key,
                    "score": round(r.score, 1),
                    "metrics": {
                        k: round(v, 2) if isinstance(v, (int, float)) else v
                        for k, v in r.metrics.items()
                    },
                }
                for r in self.results
            ],
            "insights": [i.to_dict() for i in self.insights],
            "trends": {k: v.value for k, v in self.trends.items()},
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
