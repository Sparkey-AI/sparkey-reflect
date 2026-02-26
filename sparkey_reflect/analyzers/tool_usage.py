"""
Tool Usage Analyzer

Evaluates how effectively users leverage the AI tool's capabilities:
- Tool Diversity (0-25): Variety of built-in tools used (Read, Edit, Bash, Grep, etc.)
- MCP Utilization (0-25): Whether MCP-connected tools are being used
- Slash Command Usage (0-25): Use of slash commands and shortcuts
- Automation Opportunities (0-25): Missed opportunities to use tools more effectively
"""

import re
from collections import Counter
from typing import Dict, List, Optional, Set

from sparkey_reflect.analyzers.base_analyzer import BaseReflectAnalyzer
from sparkey_reflect.core.models import (
    AnalysisResult,
    RuleFileInfo,
    Session,
)

# Known built-in tools for each AI coding tool
BUILTIN_TOOLS = {
    "claude_code": {
        "Read", "Write", "Edit", "Bash", "Glob", "Grep",
        "Task", "WebFetch", "WebSearch", "NotebookEdit",
        "AskUserQuestion", "EnterPlanMode", "ExitPlanMode",
        "TodoWrite", "TodoRead",
    },
    "cursor": {
        "codebase_search", "read_file", "edit_file", "run_terminal_command",
        "file_search", "grep_search", "list_dir", "delete_file",
    },
}

# Tools that indicate MCP usage (non-builtin, typically from .mcp.json)
MCP_TOOL_PREFIXES = ["mcp__", "mcp_"]

# Patterns in user messages that suggest slash command usage
SLASH_COMMAND_PATTERNS = [
    r"^/\w+",  # /command at start of message
    r"\b(slash command|/commit|/review|/test|/help|/clear|/compact)\b",
]

# Patterns suggesting manual work that could be automated with tools
AUTOMATION_OPPORTUNITY_PATTERNS = [
    # User pasting file contents instead of referencing
    (r"here('s| is) the (file|code|content|output)", "paste_instead_of_reference"),
    # User describing file structure instead of using glob/grep
    (r"(can you find|where is|look for|search for) .+ (file|function|class)", "manual_search"),
    # User asking to run something instead of using Bash tool directly
    (r"(can you run|please run|execute|try running)", "manual_run_request"),
    # User doing multi-step manually instead of a plan
    (r"(first .+ then .+ then|step 1.+step 2)", "manual_multi_step"),
]


class ToolUsageAnalyzer(BaseReflectAnalyzer):
    """Analyzes effectiveness of tool and command usage."""

    def get_key(self) -> str:
        return "tool_usage"

    def get_name(self) -> str:
        return "Tool Usage"

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

        # Collect tool usage across all sessions
        all_tool_names: List[str] = []
        all_mcp_tools: List[str] = []
        total_user_turns = 0
        slash_command_count = 0
        automation_misses: Dict[str, int] = Counter()
        tools_per_session: List[int] = []
        unique_tools_per_session: List[int] = []

        for session in sessions:
            session_tools: Set[str] = set()
            session_tool_count = 0

            for turn in session.turns:
                # Count tool calls from assistant turns
                if turn.tool_calls:
                    for tc in turn.tool_calls:
                        name = tc.get("name", "")
                        if name:
                            all_tool_names.append(name)
                            session_tools.add(name)
                            session_tool_count += 1
                            # Check if MCP tool
                            if any(name.startswith(p) for p in MCP_TOOL_PREFIXES):
                                all_mcp_tools.append(name)

                # Count slash commands in user turns
                if turn.role == "user" and turn.content:
                    total_user_turns += 1
                    for pattern in SLASH_COMMAND_PATTERNS:
                        if re.search(pattern, turn.content, re.IGNORECASE):
                            slash_command_count += 1
                            break

                    # Detect automation opportunities
                    for pattern, category in AUTOMATION_OPPORTUNITY_PATTERNS:
                        if re.search(pattern, turn.content, re.IGNORECASE):
                            automation_misses[category] += 1

            tools_per_session.append(session_tool_count)
            unique_tools_per_session.append(len(session_tools))

        # Calculate metrics
        tool_counter = Counter(all_tool_names)
        unique_tools_used = len(tool_counter)
        total_tool_calls = len(all_tool_names)
        mcp_tool_calls = len(all_mcp_tools)
        unique_mcp_tools = len(set(all_mcp_tools))
        total_auto_misses = sum(automation_misses.values())

        # Determine tool type for builtin comparison
        tool_type = sessions[0].tool.value if sessions else "claude_code"
        builtin_set = BUILTIN_TOOLS.get(tool_type, BUILTIN_TOOLS["claude_code"])
        builtin_used = {t for t in tool_counter if t in builtin_set}
        builtin_coverage = len(builtin_used) / len(builtin_set) if builtin_set else 0

        avg = lambda vals: sum(vals) / len(vals) if vals else 0

        # Scoring (0-25 each)
        diversity_score = self._score_diversity(unique_tools_used, builtin_coverage)
        mcp_score = self._score_mcp(mcp_tool_calls, unique_mcp_tools, rule_files)
        slash_score = self._score_slash_commands(slash_command_count, total_user_turns)
        automation_score = self._score_automation(total_auto_misses, total_user_turns)

        overall = diversity_score + mcp_score + slash_score + automation_score

        period_start = min((s.start_time for s in sessions if s.start_time), default=None)
        period_end = max((s.end_time for s in sessions if s.end_time), default=None)

        return AnalysisResult(
            analyzer_key=self.get_key(),
            analyzer_name=self.get_name(),
            score=max(0, min(100, overall)),
            metrics={
                "unique_tools_used": unique_tools_used,
                "total_tool_calls": total_tool_calls,
                "builtin_coverage": round(builtin_coverage, 3),
                "mcp_tool_calls": mcp_tool_calls,
                "unique_mcp_tools": unique_mcp_tools,
                "slash_command_count": slash_command_count,
                "slash_command_rate": round(
                    slash_command_count / total_user_turns if total_user_turns else 0, 3
                ),
                "automation_misses": total_auto_misses,
                "avg_tools_per_session": round(avg(tools_per_session), 1),
                "avg_unique_tools_per_session": round(avg(unique_tools_per_session), 1),
                "top_tools": dict(tool_counter.most_common(10)),
            },
            insights=[],
            session_count=len(sessions),
            period_start=period_start,
            period_end=period_end,
        )

    # =========================================================================
    # Scoring Dimensions (0-25 each)
    # =========================================================================

    def _score_diversity(self, unique_tools: int, builtin_coverage: float) -> float:
        """Score 0-25: Variety of tools used."""
        score = 0.0
        # Unique tool count
        if unique_tools >= 8:
            score += 15
        elif unique_tools >= 5:
            score += 11
        elif unique_tools >= 3:
            score += 7
        elif unique_tools >= 1:
            score += 3

        # Builtin coverage bonus
        if builtin_coverage >= 0.5:
            score += 10
        elif builtin_coverage >= 0.3:
            score += 7
        elif builtin_coverage >= 0.15:
            score += 4

        return min(25, score)

    def _score_mcp(
        self, mcp_calls: int, unique_mcp: int,
        rule_files: Optional[List[RuleFileInfo]],
    ) -> float:
        """Score 0-25: MCP tool utilization."""
        # Check if MCP is even configured
        has_mcp_config = False
        if rule_files:
            for rf in rule_files:
                if rf.file_type in ("mcp_config", "claude_user_mcp") and rf.exists:
                    has_mcp_config = True
                    break

        if not has_mcp_config:
            # No MCP configured -- give neutral score (not penalized)
            return 15

        # MCP is configured, score based on usage
        if unique_mcp >= 3:
            return 25
        if unique_mcp >= 2:
            return 22
        if unique_mcp >= 1:
            return 18
        # Configured but never used
        return 8

    def _score_slash_commands(self, count: int, total_turns: int) -> float:
        """Score 0-25: Slash command and shortcut usage."""
        if total_turns == 0:
            return 0
        rate = count / total_turns
        if rate >= 0.1:
            return 25
        if rate >= 0.05:
            return 20
        if rate >= 0.02:
            return 15
        if count >= 1:
            return 10
        return 5  # not penalized heavily since not all users need them

    def _score_automation(self, misses: int, total_turns: int) -> float:
        """Score 0-25: Fewer automation misses = better."""
        if total_turns == 0:
            return 15
        miss_rate = misses / total_turns
        if miss_rate <= 0.02:
            return 25
        if miss_rate <= 0.05:
            return 20
        if miss_rate <= 0.1:
            return 15
        if miss_rate <= 0.2:
            return 10
        return 5

