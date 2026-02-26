"""
Claude Code Reader

Reads conversation data from Claude Code's local storage:
- ~/.claude/projects/<project-hash>/<session-uuid>.jsonl  (JSONL per session)
- CLAUDE.md, .claude/settings.json, .mcp.json, etc.  (rule/config files)

Each JSONL file is one conversation session. Each line is a message entry with:
  - type: "user" | "assistant" | "file-history-snapshot" | "tool_result" | etc.
  - message: {role, content, model, usage}
  - timestamp: ISO8601
  - sessionId: UUID
  - cwd: workspace path
  - gitBranch: current branch
"""

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from sparkey_reflect.core.models import (
    ConversationTurn,
    RuleFileInfo,
    Session,
    SessionType,
    ToolType,
)
from sparkey_reflect.readers.base_reader import BaseReader

logger = logging.getLogger(__name__)

CLAUDE_DIR = Path.home() / ".claude"
PROJECTS_DIR = CLAUDE_DIR / "projects"

# Patterns for classifying session type from conversation content
SESSION_TYPE_PATTERNS = {
    SessionType.DEBUGGING: [
        r"\b(debug|error|traceback|exception|fix|bug|issue|broken|crash|fail)\b",
    ],
    SessionType.TESTING: [
        r"\b(test|spec|assert|mock|fixture|coverage|pytest|jest|unittest)\b",
    ],
    SessionType.REFACTORING: [
        r"\b(refactor|rename|extract|restructure|reorganize|clean.?up|simplif)\b",
    ],
    SessionType.DOCS: [
        r"\b(document|readme|docstring|comment|explain|description|api.?doc)\b",
    ],
    SessionType.EXPLORATION: [
        r"\b(explore|search|find|where|how does|what is|understand|investigate)\b",
    ],
}


class ClaudeCodeReader(BaseReader):
    """Reader for Claude Code conversation data."""

    def get_tool_type(self) -> ToolType:
        return ToolType.CLAUDE_CODE

    def is_available(self) -> bool:
        if not PROJECTS_DIR.exists():
            return False
        # Check if any project dir has JSONL files
        for project_dir in PROJECTS_DIR.iterdir():
            if project_dir.is_dir():
                if any(project_dir.glob("*.jsonl")):
                    return True
        return False

    def get_data_locations(self) -> List[str]:
        locations = []
        if PROJECTS_DIR.exists():
            for project_dir in PROJECTS_DIR.iterdir():
                if project_dir.is_dir() and any(project_dir.glob("*.jsonl")):
                    locations.append(str(project_dir))
        return locations

    def get_history_range(self) -> Optional[Tuple[datetime, datetime]]:
        earliest = None
        latest = None
        for jsonl_file in self._iter_session_files():
            try:
                mtime = datetime.fromtimestamp(
                    jsonl_file.stat().st_mtime, tz=timezone.utc
                )
                ctime = datetime.fromtimestamp(
                    jsonl_file.stat().st_ctime, tz=timezone.utc
                )
                ts = min(mtime, ctime)
                if earliest is None or ts < earliest:
                    earliest = ts
                if latest is None or mtime > latest:
                    latest = mtime
            except OSError:
                continue
        if earliest and latest:
            return (earliest, latest)
        return None

    # =========================================================================
    # Session Reading
    # =========================================================================

    def read_sessions(
        self,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        workspace_path: Optional[str] = None,
    ) -> List[Session]:
        sessions = []

        for jsonl_file in self._iter_session_files():
            try:
                # Quick date filter based on file mtime
                mtime = datetime.fromtimestamp(
                    jsonl_file.stat().st_mtime, tz=timezone.utc
                )
                if since:
                    since_aware = since if since.tzinfo else since.replace(tzinfo=timezone.utc)
                    if mtime < since_aware:
                        continue

                session = self._parse_jsonl_session(jsonl_file)
                if session is None:
                    continue

                # Apply time filters on parsed session timestamps
                if since and session.start_time:
                    s_start = session.start_time if session.start_time.tzinfo else session.start_time.replace(tzinfo=timezone.utc)
                    since_aware = since if since.tzinfo else since.replace(tzinfo=timezone.utc)
                    if s_start < since_aware:
                        continue
                if until and session.start_time:
                    s_start = session.start_time if session.start_time.tzinfo else session.start_time.replace(tzinfo=timezone.utc)
                    until_aware = until if until.tzinfo else until.replace(tzinfo=timezone.utc)
                    if s_start > until_aware:
                        continue

                # Apply workspace filter
                if workspace_path and session.workspace_path:
                    if workspace_path not in session.workspace_path:
                        continue

                sessions.append(session)

            except Exception as e:
                logger.warning("Error reading %s: %s", jsonl_file, e)
                continue

        sessions.sort(key=lambda s: s.start_time or datetime.min.replace(tzinfo=timezone.utc))
        return sessions

    def _iter_session_files(self):
        """Yield all session JSONL files across all projects."""
        if not PROJECTS_DIR.exists():
            return
        for project_dir in PROJECTS_DIR.iterdir():
            if not project_dir.is_dir():
                continue
            for jsonl_file in project_dir.glob("*.jsonl"):
                yield jsonl_file

    def _parse_jsonl_session(self, file_path: Path) -> Optional[Session]:
        """Parse a Claude Code JSONL session file into a Session."""
        turns = []
        total_input = 0
        total_output = 0
        first_ts = None
        last_ts = None
        workspace = None
        branch = None
        model = None
        session_id = None

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    entry_type = entry.get("type", "")

                    # Skip non-message entries
                    if entry_type in ("file-history-snapshot", "summary"):
                        continue

                    # Extract session metadata from first entry
                    if session_id is None:
                        session_id = entry.get("sessionId")
                    if workspace is None:
                        workspace = entry.get("cwd")
                    if branch is None:
                        branch = entry.get("gitBranch")

                    # Parse the message
                    msg_data = entry.get("message", {})
                    if not isinstance(msg_data, dict):
                        continue

                    role = msg_data.get("role") or entry_type
                    if role not in ("user", "assistant", "system", "tool", "tool_result"):
                        continue

                    # Extract model from assistant messages
                    if msg_data.get("model"):
                        model = msg_data["model"]

                    # Parse turn
                    turn = self._parse_turn(msg_data, entry)
                    if turn is None:
                        continue

                    turns.append(turn)
                    total_input += turn.input_tokens
                    total_output += turn.output_tokens

                    if turn.timestamp:
                        if first_ts is None or turn.timestamp < first_ts:
                            first_ts = turn.timestamp
                        if last_ts is None or turn.timestamp > last_ts:
                            last_ts = turn.timestamp

        except OSError as e:
            logger.debug("Cannot read %s: %s", file_path, e)
            return None

        if not turns:
            return None

        # Use file stem as session ID fallback
        if not session_id:
            session_id = file_path.stem

        # Calculate duration
        duration = 0.0
        if first_ts and last_ts:
            duration = (last_ts - first_ts).total_seconds() / 60.0

        # Resolve workspace from project dir name if not in entries
        if not workspace:
            workspace = self._resolve_workspace_from_dir(file_path.parent)

        session_type = self._classify_session(turns)

        return Session(
            session_id=f"cc_{session_id}",
            tool=ToolType.CLAUDE_CODE,
            turns=turns,
            start_time=first_ts,
            end_time=last_ts,
            duration_minutes=duration,
            workspace_path=workspace,
            branch=branch,
            model=model,
            total_input_tokens=total_input,
            total_output_tokens=total_output,
            session_type=session_type,
            metadata={
                "file_path": str(file_path),
                "project_dir": str(file_path.parent),
            },
        )

    def _parse_turn(self, msg: Dict[str, Any], entry: Dict[str, Any]) -> Optional[ConversationTurn]:
        """Parse a message dict and its JSONL entry into a ConversationTurn."""
        role = msg.get("role", entry.get("type", ""))
        if role not in ("user", "assistant", "system", "tool", "tool_result"):
            return None

        # Extract content
        content = ""
        tool_calls = []
        tool_name = None
        has_error = False
        has_code = False
        file_refs = []

        raw_content = msg.get("content", "")
        if isinstance(raw_content, str):
            content = raw_content
        elif isinstance(raw_content, list):
            # Claude API content blocks
            text_parts = []
            for block in raw_content:
                if not isinstance(block, dict):
                    text_parts.append(str(block))
                    continue
                block_type = block.get("type", "")
                if block_type == "text":
                    text_parts.append(block.get("text", ""))
                elif block_type == "tool_use":
                    tool_calls.append({
                        "name": block.get("name", ""),
                        "id": block.get("id", ""),
                    })
                elif block_type == "tool_result":
                    tool_name = block.get("tool_use_id", "")
                    sub_content = block.get("content", "")
                    if isinstance(sub_content, str):
                        text_parts.append(sub_content)
                    elif isinstance(sub_content, list):
                        for sub in sub_content:
                            if isinstance(sub, dict) and sub.get("type") == "text":
                                text_parts.append(sub.get("text", ""))
            content = "\n".join(p for p in text_parts if p)

        # Detect error context
        if content:
            error_patterns = r"(error|exception|traceback|stack trace|failed|errno)"
            if re.search(error_patterns, content, re.IGNORECASE):
                has_error = True

            # Detect code snippets
            if "```" in content or content.count("\n") > 3:
                has_code = True

            # Extract file references
            file_pattern = r'[\w./\\-]+\.\w{1,10}'
            file_refs = list(set(re.findall(file_pattern, content)))[:20]

        # Parse timestamp from JSONL entry (not from message)
        timestamp = None
        ts_str = entry.get("timestamp") or msg.get("timestamp")
        if ts_str:
            timestamp = self._parse_timestamp(ts_str)

        # Token counts from usage block
        usage = msg.get("usage", {})
        input_tokens = 0
        output_tokens = 0
        if isinstance(usage, dict):
            input_tokens = usage.get("input_tokens", 0) or 0
            # Include cache tokens in input count
            input_tokens += usage.get("cache_read_input_tokens", 0) or 0
            input_tokens += usage.get("cache_creation_input_tokens", 0) or 0
            output_tokens = usage.get("output_tokens", 0) or 0

        # Map tool role
        if role == "tool":
            role = "tool_result"

        return ConversationTurn(
            role=role,
            content=content,
            timestamp=timestamp,
            tool_calls=tool_calls,
            tool_name=tool_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            file_references=file_refs,
            has_error_context=has_error,
            has_code_snippet=has_code,
        )

    def _parse_timestamp(self, ts) -> Optional[datetime]:
        """Parse various timestamp formats."""
        if isinstance(ts, (int, float)):
            if ts > 1e12:
                ts = ts / 1000
            try:
                return datetime.fromtimestamp(ts, tz=timezone.utc)
            except (ValueError, OSError):
                return None
        if isinstance(ts, str):
            # Try fromisoformat first (handles most Claude Code timestamps)
            try:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
            except ValueError:
                pass
            # Fallback formats
            for fmt in (
                "%Y-%m-%dT%H:%M:%S.%fZ",
                "%Y-%m-%dT%H:%M:%SZ",
                "%Y-%m-%dT%H:%M:%S.%f%z",
                "%Y-%m-%dT%H:%M:%S%z",
            ):
                try:
                    dt = datetime.strptime(ts, fmt)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    return dt
                except ValueError:
                    continue
        return None

    def _resolve_workspace_from_dir(self, project_dir: Path) -> Optional[str]:
        """Resolve project dir name to an actual workspace path.

        Claude Code uses the format: -Users-user-Dev-project
        which maps to /Users/user/Dev/project
        """
        dir_name = project_dir.name
        # Convert dash-separated path back to filesystem path
        if dir_name.startswith("-"):
            return "/" + dir_name[1:].replace("-", "/")
        return dir_name

    def _classify_session(self, turns: List[ConversationTurn]) -> SessionType:
        """Classify session type based on user message content."""
        user_text = " ".join(
            t.content for t in turns if t.role == "user" and t.content
        ).lower()

        if not user_text:
            return SessionType.UNKNOWN

        scores: Dict[SessionType, int] = {}
        for stype, patterns in SESSION_TYPE_PATTERNS.items():
            count = 0
            for pat in patterns:
                count += len(re.findall(pat, user_text, re.IGNORECASE))
            if count > 0:
                scores[stype] = count

        if scores:
            return max(scores, key=scores.get)

        return SessionType.CODING

    # =========================================================================
    # Rule File Reading
    # =========================================================================

    def read_rule_files(
        self,
        workspace_path: Optional[str] = None,
    ) -> List[RuleFileInfo]:
        workspace = Path(workspace_path) if workspace_path else Path.cwd()
        rule_files = []

        # Project-level files
        project_files = [
            ("CLAUDE.md", "claude_md"),
            (".claude/settings.json", "claude_settings"),
            (".claude/settings.local.json", "claude_settings_local"),
            (".mcp.json", "mcp_config"),
            (".claudeignore", "claudeignore"),
            (".claude/hooks.json", "claude_hooks"),
        ]
        for rel_path, file_type in project_files:
            full_path = workspace / rel_path
            rule_files.append(self._read_rule_file(full_path, file_type))

        # Nested CLAUDE.md files (search one level deep)
        try:
            for subdir in workspace.iterdir():
                if subdir.is_dir() and not subdir.name.startswith("."):
                    nested = subdir / "CLAUDE.md"
                    if nested.exists():
                        rule_files.append(
                            self._read_rule_file(nested, "claude_md_nested")
                        )
        except OSError:
            pass

        # User-level files
        user_files = [
            (CLAUDE_DIR / "settings.json", "claude_user_settings"),
            (Path.home() / ".claude.json", "claude_user_mcp"),
        ]
        for path, file_type in user_files:
            rule_files.append(self._read_rule_file(path, file_type))

        # Memory files
        if PROJECTS_DIR.exists():
            for project_dir in PROJECTS_DIR.iterdir():
                if not project_dir.is_dir():
                    continue
                memory_dir = project_dir / "memory"
                if memory_dir.exists():
                    memory_md = memory_dir / "MEMORY.md"
                    if memory_md.exists():
                        rule_files.append(
                            self._read_rule_file(memory_md, "claude_memory")
                        )

        return rule_files

    def _read_rule_file(self, path: Path, file_type: str) -> RuleFileInfo:
        """Read and analyze a single rule/config file."""
        info = RuleFileInfo(
            file_path=str(path),
            file_type=file_type,
            tool=ToolType.CLAUDE_CODE,
            exists=path.exists(),
        )

        if not path.exists():
            return info

        try:
            content = path.read_text(encoding="utf-8")
            info.raw_content = content
            info.word_count = len(content.split())
            info.last_modified = datetime.fromtimestamp(
                path.stat().st_mtime, tz=timezone.utc
            )

            if file_type.startswith("claude_md") or file_type == "claude_memory":
                self._analyze_markdown_file(info, content)
            elif file_type in ("claude_settings", "claude_settings_local",
                               "mcp_config", "claude_user_settings",
                               "claude_user_mcp"):
                info.has_tool_config = True
                try:
                    data = json.loads(content)
                    info.section_count = len(data) if isinstance(data, dict) else 0
                except json.JSONDecodeError:
                    pass

        except OSError as e:
            logger.debug("Cannot read %s: %s", path, e)
            info.exists = False

        return info

    def _analyze_markdown_file(self, info: RuleFileInfo, content: str):
        """Analyze a markdown instruction file for quality signals."""
        lines = content.split("\n")

        # Count sections (headers)
        sections = [line.lstrip("#").strip() for line in lines if line.startswith("#")]
        info.sections = sections
        info.section_count = len(sections)

        lower = content.lower()
        info.has_examples = any(
            kw in lower for kw in ["example", "```", "e.g.", "for instance"]
        )
        info.has_constraints = any(
            kw in lower
            for kw in [
                "never", "always", "must", "don't", "do not",
                "avoid", "prefer", "required", "forbidden",
            ]
        )
        info.has_project_context = any(
            kw in lower
            for kw in [
                "project", "architecture", "stack", "framework",
                "directory", "structure", "overview",
            ]
        )
        info.has_style_guide = any(
            kw in lower
            for kw in [
                "style", "naming", "convention", "format",
                "lint", "indent", "camelcase", "snake_case",
            ]
        )
