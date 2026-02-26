"""
Tool Usage Analyzer

Evaluates how effectively users leverage the AI tool's capabilities across
five dimensions using smooth scoring curves:
- Diversity (w=0.25): Variety of built-in tools used
- MCP Utilization (w=0.15): Whether MCP-connected tools are being used
- Slash Commands (w=0.15): Use of slash commands and shortcuts
- Automation Opportunities (w=0.20): Missed opportunities to use tools effectively
- Tool Appropriateness (w=0.25): Using the right tool for the job (NEW)

Benchmarks: Broader tool use correlates with higher skill level. Using
specialized tools (Edit vs Bash sed) indicates mastery.
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
from sparkey_reflect.core.scoring import count_score, diminishing, sigmoid, weighted_sum

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
    (r"here('s| is) the (file|code|content|output)", "paste_instead_of_reference"),
    (r"(can you find|where is|look for|search for) .+ (file|function|class)", "manual_search"),
    (r"(can you run|please run|execute|try running)", "manual_run_request"),
    (r"(first .+ then .+ then|step 1.+step 2)", "manual_multi_step"),
]

# Specialized tools that SHOULD be used instead of Bash for specific tasks
APPROPRIATE_FILE_TOOLS = {"Edit", "Write", "edit_file"}
INAPPROPRIATE_FILE_BASH_PATTERNS = [
    r"\bsed\b", r"\bawk\b", r"\becho\s+.*>", r"\bcat\s+<<",
]
APPROPRIATE_READ_TOOLS = {"Read", "read_file"}
INAPPROPRIATE_READ_BASH_PATTERNS = [
    r"\bcat\b", r"\bhead\b", r"\btail\b",
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

        # Tool appropriateness tracking
        file_mod_appropriate = 0
        file_mod_inappropriate = 0
        file_read_appropriate = 0
        file_read_inappropriate = 0

        for session in sessions:
            session_tools: Set[str] = set()
            session_tool_count = 0

            for turn in session.turns:
                if turn.tool_calls:
                    for tc in turn.tool_calls:
                        name = tc.get("name", "")
                        if name:
                            all_tool_names.append(name)
                            session_tools.add(name)
                            session_tool_count += 1
                            if any(name.startswith(p) for p in MCP_TOOL_PREFIXES):
                                all_mcp_tools.append(name)

                            # Track tool appropriateness
                            if name in APPROPRIATE_FILE_TOOLS:
                                file_mod_appropriate += 1
                            if name in APPROPRIATE_READ_TOOLS:
                                file_read_appropriate += 1
                            if name == "Bash" or name == "run_terminal_command":
                                args = tc.get("arguments", tc.get("input", ""))
                                cmd = args if isinstance(args, str) else str(args)
                                for pattern in INAPPROPRIATE_FILE_BASH_PATTERNS:
                                    if re.search(pattern, cmd):
                                        file_mod_inappropriate += 1
                                        break
                                for pattern in INAPPROPRIATE_READ_BASH_PATTERNS:
                                    if re.search(pattern, cmd):
                                        file_read_inappropriate += 1
                                        break

                if turn.role == "user" and turn.content:
                    total_user_turns += 1
                    for pattern in SLASH_COMMAND_PATTERNS:
                        if re.search(pattern, turn.content, re.IGNORECASE):
                            slash_command_count += 1
                            break

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

        tool_type = sessions[0].tool.value if sessions else "claude_code"
        builtin_set = BUILTIN_TOOLS.get(tool_type, BUILTIN_TOOLS["claude_code"])
        builtin_used = {t for t in tool_counter if t in builtin_set}
        builtin_coverage = len(builtin_used) / len(builtin_set) if builtin_set else 0

        miss_rate = total_auto_misses / total_user_turns if total_user_turns > 0 else 0
        slash_rate = slash_command_count / total_user_turns if total_user_turns > 0 else 0

        # Tool appropriateness ratio
        total_file_ops = file_mod_appropriate + file_mod_inappropriate
        total_read_ops = file_read_appropriate + file_read_inappropriate
        if total_file_ops + total_read_ops > 0:
            appropriateness = (
                (file_mod_appropriate + file_read_appropriate) /
                (total_file_ops + total_read_ops)
            )
        else:
            appropriateness = 0.7  # neutral if no file operations detected

        avg = lambda vals: sum(vals) / len(vals) if vals else 0

        # Smooth scoring: each dimension 0-1
        diversity_dim = (
            diminishing(unique_tools_used, 8) * 0.6
            + sigmoid(builtin_coverage, 0.3, 5) * 0.4
        )

        # MCP: neutral if unconfigured
        has_mcp_config = False
        if rule_files:
            for rf in rule_files:
                if rf.file_type in ("mcp_config", "claude_user_mcp") and rf.exists:
                    has_mcp_config = True
                    break
        if not has_mcp_config:
            mcp_dim = 0.6  # neutral (15/25 equivalent)
        else:
            mcp_dim = count_score(unique_mcp_tools, [
                (0, 0.3), (1, 0.7), (2, 0.85), (3, 1.0),
            ])

        slash_dim = sigmoid(slash_rate, 0.05, 30)
        automation_dim = 1 - sigmoid(miss_rate, 0.08, 15)
        appropriateness_dim = sigmoid(appropriateness, 0.6, 4)

        overall = weighted_sum([
            (diversity_dim, 0.25),
            (mcp_dim, 0.15),
            (slash_dim, 0.15),
            (automation_dim, 0.20),
            (appropriateness_dim, 0.25),
        ])

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
                "slash_command_rate": round(slash_rate, 3),
                "automation_misses": total_auto_misses,
                "tool_appropriateness": round(appropriateness, 3),
                "avg_tools_per_session": round(avg(tools_per_session), 1),
                "avg_unique_tools_per_session": round(avg(unique_tools_per_session), 1),
                "top_tools": dict(tool_counter.most_common(10)),
            },
            insights=[],
            session_count=len(sessions),
            period_start=period_start,
            period_end=period_end,
        )
