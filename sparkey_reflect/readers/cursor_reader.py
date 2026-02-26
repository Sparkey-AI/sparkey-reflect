"""
Cursor Reader

Reads conversation data from Cursor IDE's local SQLite storage:
- ~/Library/Application Support/Cursor/User/workspaceStorage/<hash>/state.vscdb
- .cursorrules, .cursor/rules/*.mdc, .cursor/mcp.json, etc.

Each state.vscdb is a SQLite database with table `cursorDiskKV`:
  - composer.composerData: AI composition sessions with turns
  - aiService.prompts: User prompts
  - aiService.generations: AI-generated completions
"""

import json
import logging
import platform
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from sparkey_reflect.core.models import (
    ConversationTurn,
    RuleFileInfo,
    Session,
    SessionType,
    ToolType,
)
from sparkey_reflect.readers.base_reader import BaseReader

logger = logging.getLogger(__name__)

# Platform-specific Cursor storage directories
if platform.system() == "Darwin":
    CURSOR_BASE_DIR = Path.home() / "Library" / "Application Support" / "Cursor"
elif platform.system() == "Windows":
    CURSOR_BASE_DIR = Path.home() / "AppData" / "Roaming" / "Cursor"
else:
    CURSOR_BASE_DIR = Path.home() / ".config" / "Cursor"

WORKSPACE_STORAGE_DIR = CURSOR_BASE_DIR / "User" / "workspaceStorage"

# Keys to extract from cursorDiskKV
TARGET_KEYS = [
    "composer.composerData",
    "aiService.prompts",
    "aiService.generations",
]

SQLITE_TIMEOUT = 5

# Session type classification patterns (shared with claude_code_reader)
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


class CursorReader(BaseReader):
    """Reader for Cursor IDE conversation data."""

    def get_tool_type(self) -> ToolType:
        return ToolType.CURSOR

    def is_available(self) -> bool:
        if not WORKSPACE_STORAGE_DIR.exists():
            return False
        return bool(self._find_db_files())

    def get_data_locations(self) -> List[str]:
        return [str(f) for f in self._find_db_files()]

    def get_history_range(self) -> Optional[Tuple[datetime, datetime]]:
        earliest = None
        latest = None
        for db_file in self._find_db_files():
            try:
                mtime = datetime.fromtimestamp(
                    db_file.stat().st_mtime, tz=timezone.utc
                )
                ctime = datetime.fromtimestamp(
                    db_file.stat().st_ctime, tz=timezone.utc
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

        for db_file in self._find_db_files():
            # Quick date filter on file mtime
            if since:
                try:
                    mtime = datetime.fromtimestamp(
                        db_file.stat().st_mtime, tz=timezone.utc
                    )
                    since_aware = since if since.tzinfo else since.replace(tzinfo=timezone.utc)
                    if mtime < since_aware:
                        continue
                except OSError:
                    continue

            try:
                db_sessions = self._parse_db_file(db_file, since, until, workspace_path)
                sessions.extend(db_sessions)
            except Exception as e:
                logger.warning("Error reading Cursor DB %s: %s", db_file, e)

        sessions.sort(key=lambda s: s.start_time or datetime.min.replace(tzinfo=timezone.utc))
        return sessions

    def _find_db_files(self) -> List[Path]:
        """Find all state.vscdb files in Cursor workspace storage."""
        if not WORKSPACE_STORAGE_DIR.exists():
            return []
        files = list(WORKSPACE_STORAGE_DIR.glob("*/state.vscdb"))
        # Sort by modification time (newest first)
        files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        return files

    def _parse_db_file(
        self,
        db_path: Path,
        since: Optional[datetime],
        until: Optional[datetime],
        workspace_filter: Optional[str],
    ) -> List[Session]:
        """Parse a single state.vscdb file into Sessions."""
        workspace_hash = db_path.parent.name

        conn = None
        try:
            uri = f"file:{db_path}?mode=ro"
            conn = sqlite3.connect(uri, uri=True, timeout=SQLITE_TIMEOUT)
            conn.row_factory = sqlite3.Row

            cursor = conn.cursor()
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='cursorDiskKV'"
            )
            if cursor.fetchone() is None:
                return []

            placeholders = ",".join(["?"] * len(TARGET_KEYS))
            cursor.execute(
                f"SELECT key, value FROM cursorDiskKV WHERE key IN ({placeholders})",
                TARGET_KEYS,
            )

            # Collect raw records by source key
            composer_records = []
            prompt_records = []
            generation_records = []

            for row in cursor.fetchall():
                key = row["key"]
                raw_value = row["value"]
                if not raw_value:
                    continue
                try:
                    data = json.loads(raw_value)
                except (json.JSONDecodeError, TypeError):
                    continue

                if key == "composer.composerData":
                    composer_records = self._parse_composer_data(data)
                elif key == "aiService.prompts":
                    prompt_records = self._parse_prompts(data)
                elif key == "aiService.generations":
                    generation_records = self._parse_generations(data)

        except (sqlite3.OperationalError, sqlite3.DatabaseError) as e:
            logger.warning("SQLite error for %s: %s", db_path, e)
            return []
        finally:
            if conn:
                conn.close()

        # Build sessions from composer records (primary)
        # Fall back to pairing prompts + generations if no composer data
        sessions = []

        if composer_records:
            sessions = self._build_sessions_from_composer(
                composer_records, workspace_hash, str(db_path)
            )
        else:
            sessions = self._build_sessions_from_prompt_gen(
                prompt_records, generation_records, workspace_hash, str(db_path)
            )

        # Apply time and workspace filters
        filtered = []
        for session in sessions:
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
            if workspace_filter and session.workspace_path:
                if workspace_filter not in session.workspace_path:
                    continue
            filtered.append(session)

        return filtered

    # =========================================================================
    # Composer Data Parsing
    # =========================================================================

    def _parse_composer_data(self, data: Any) -> List[Dict[str, Any]]:
        """Parse composer.composerData JSON into raw session records."""
        records = []

        items = []
        if isinstance(data, dict):
            for session_id, session_data in data.items():
                if isinstance(session_data, dict):
                    session_data["_session_id"] = session_id
                    items.append(session_data)
                elif isinstance(session_data, list):
                    for entry in session_data:
                        if isinstance(entry, dict):
                            entry["_session_id"] = session_id
                            items.append(entry)
        elif isinstance(data, list):
            items = [item for item in data if isinstance(item, dict)]

        for item in items:
            record = {
                "session_id": item.get("_session_id") or item.get("composerId") or item.get("id"),
                "timestamp": self._extract_timestamp(item),
                "model": item.get("model") or item.get("modelId"),
                "prompt": item.get("prompt") or item.get("text") or item.get("input"),
                "completion": item.get("completion") or item.get("response") or item.get("output"),
                "input_tokens": self._extract_token_count(item, "input"),
                "output_tokens": self._extract_token_count(item, "output"),
                "turns": item.get("conversation") or item.get("turns") or item.get("messages") or [],
                "source_key": "composer.composerData",
            }
            records.append(record)

        return records

    def _parse_prompts(self, data: Any) -> List[Dict[str, Any]]:
        """Parse aiService.prompts JSON."""
        records = []
        items = data if isinstance(data, list) else [data] if isinstance(data, dict) else []
        for item in items:
            if not isinstance(item, dict):
                continue
            records.append({
                "session_id": item.get("sessionId") or item.get("id"),
                "timestamp": self._extract_timestamp(item),
                "model": item.get("model") or item.get("modelId"),
                "prompt": item.get("prompt") or item.get("text") or item.get("content"),
                "completion": None,
                "input_tokens": self._extract_token_count(item, "input"),
                "output_tokens": 0,
                "source_key": "aiService.prompts",
            })
        return records

    def _parse_generations(self, data: Any) -> List[Dict[str, Any]]:
        """Parse aiService.generations JSON."""
        records = []
        items = data if isinstance(data, list) else [data] if isinstance(data, dict) else []
        for item in items:
            if not isinstance(item, dict):
                continue
            records.append({
                "session_id": item.get("sessionId") or item.get("id"),
                "timestamp": self._extract_timestamp(item),
                "model": item.get("model") or item.get("modelId"),
                "prompt": item.get("prompt") or item.get("input"),
                "completion": item.get("completion") or item.get("text") or item.get("output"),
                "input_tokens": self._extract_token_count(item, "input"),
                "output_tokens": self._extract_token_count(item, "output"),
                "source_key": "aiService.generations",
            })
        return records

    # =========================================================================
    # Session Building
    # =========================================================================

    def _build_sessions_from_composer(
        self, records: List[Dict], workspace_hash: str, source_file: str,
    ) -> List[Session]:
        """Build Session objects from composer data records."""
        sessions = []

        for record in records:
            session_id = record.get("session_id") or "unknown"
            turns = []
            model = record.get("model")
            total_input = record.get("input_tokens", 0) or 0
            total_output = record.get("output_tokens", 0) or 0

            # Parse embedded conversation turns
            raw_turns = record.get("turns", [])
            if isinstance(raw_turns, list):
                for raw_turn in raw_turns:
                    if not isinstance(raw_turn, dict):
                        continue
                    turn = self._parse_raw_turn(raw_turn)
                    if turn:
                        turns.append(turn)
                        total_input += turn.input_tokens
                        total_output += turn.output_tokens

            # If no embedded turns, create turns from prompt/completion
            if not turns:
                prompt = record.get("prompt")
                completion = record.get("completion")
                if prompt:
                    turns.append(ConversationTurn(
                        role="user",
                        content=str(prompt),
                        timestamp=record.get("timestamp"),
                        file_references=self._extract_file_refs(str(prompt)),
                        has_error_context=self._has_error(str(prompt)),
                        has_code_snippet="```" in str(prompt),
                    ))
                if completion:
                    turns.append(ConversationTurn(
                        role="assistant",
                        content=str(completion),
                        timestamp=record.get("timestamp"),
                    ))

            if not turns:
                continue

            # Compute timestamps
            timestamps = [t.timestamp for t in turns if t.timestamp]
            start_time = record.get("timestamp") or (min(timestamps) if timestamps else None)
            end_time = max(timestamps) if timestamps else start_time

            duration = 0.0
            if start_time and end_time:
                try:
                    duration = (end_time - start_time).total_seconds() / 60.0
                except (TypeError, AttributeError):
                    pass

            session_type = self._classify_session(turns)

            sessions.append(Session(
                session_id=f"cur_{session_id}",
                tool=ToolType.CURSOR,
                turns=turns,
                start_time=start_time,
                end_time=end_time,
                duration_minutes=max(0, duration),
                workspace_path=workspace_hash,
                model=model,
                total_input_tokens=total_input,
                total_output_tokens=total_output,
                session_type=session_type,
                metadata={
                    "source_file": source_file,
                    "source_key": "composer.composerData",
                    "workspace_hash": workspace_hash,
                },
            ))

        return sessions

    def _build_sessions_from_prompt_gen(
        self,
        prompts: List[Dict],
        generations: List[Dict],
        workspace_hash: str,
        source_file: str,
    ) -> List[Session]:
        """Build sessions by pairing prompt and generation records."""
        # Group by session_id
        session_map: Dict[str, List[Dict]] = {}
        for record in prompts + generations:
            sid = record.get("session_id") or "unknown"
            session_map.setdefault(sid, []).append(record)

        sessions = []
        for sid, records in session_map.items():
            turns = []
            total_input = 0
            total_output = 0
            model = None

            # Sort by timestamp
            records.sort(key=lambda r: r.get("timestamp") or datetime.min.replace(tzinfo=timezone.utc))

            for record in records:
                if not model and record.get("model"):
                    model = record["model"]

                prompt = record.get("prompt")
                completion = record.get("completion")

                if prompt:
                    turns.append(ConversationTurn(
                        role="user",
                        content=str(prompt),
                        timestamp=record.get("timestamp"),
                        input_tokens=record.get("input_tokens", 0) or 0,
                        file_references=self._extract_file_refs(str(prompt)),
                        has_error_context=self._has_error(str(prompt)),
                        has_code_snippet="```" in str(prompt),
                    ))
                    total_input += record.get("input_tokens", 0) or 0

                if completion:
                    turns.append(ConversationTurn(
                        role="assistant",
                        content=str(completion),
                        timestamp=record.get("timestamp"),
                        output_tokens=record.get("output_tokens", 0) or 0,
                    ))
                    total_output += record.get("output_tokens", 0) or 0

            if not turns:
                continue

            timestamps = [t.timestamp for t in turns if t.timestamp]
            start_time = min(timestamps) if timestamps else None
            end_time = max(timestamps) if timestamps else None

            duration = 0.0
            if start_time and end_time:
                try:
                    duration = (end_time - start_time).total_seconds() / 60.0
                except (TypeError, AttributeError):
                    pass

            sessions.append(Session(
                session_id=f"cur_{sid}",
                tool=ToolType.CURSOR,
                turns=turns,
                start_time=start_time,
                end_time=end_time,
                duration_minutes=max(0, duration),
                workspace_path=workspace_hash,
                model=model,
                total_input_tokens=total_input,
                total_output_tokens=total_output,
                session_type=self._classify_session(turns),
                metadata={
                    "source_file": source_file,
                    "source_key": "aiService.prompts+generations",
                    "workspace_hash": workspace_hash,
                },
            ))

        return sessions

    # =========================================================================
    # Helpers
    # =========================================================================

    def _parse_raw_turn(self, raw: Dict[str, Any]) -> Optional[ConversationTurn]:
        """Parse a raw conversation turn dict from composer data."""
        role = raw.get("role") or raw.get("type", "")
        if role not in ("user", "assistant", "system", "tool", "tool_result", "human", "ai"):
            return None

        # Normalize roles
        if role == "human":
            role = "user"
        elif role == "ai":
            role = "assistant"
        elif role == "tool":
            role = "tool_result"

        content = ""
        raw_content = raw.get("content") or raw.get("text") or raw.get("message") or ""
        if isinstance(raw_content, str):
            content = raw_content
        elif isinstance(raw_content, list):
            text_parts = []
            for block in raw_content:
                if isinstance(block, str):
                    text_parts.append(block)
                elif isinstance(block, dict):
                    text_parts.append(block.get("text", ""))
            content = "\n".join(p for p in text_parts if p)

        # Extract tool calls
        tool_calls = []
        if raw.get("tool_calls"):
            for tc in raw["tool_calls"]:
                if isinstance(tc, dict):
                    tool_calls.append({
                        "name": tc.get("name") or tc.get("function", {}).get("name", ""),
                        "id": tc.get("id", ""),
                    })

        timestamp = self._extract_timestamp(raw)
        input_tokens = self._extract_token_count(raw, "input")
        output_tokens = self._extract_token_count(raw, "output")

        return ConversationTurn(
            role=role,
            content=content,
            timestamp=timestamp,
            tool_calls=tool_calls,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            file_references=self._extract_file_refs(content) if role == "user" else [],
            has_error_context=self._has_error(content) if role == "user" else False,
            has_code_snippet="```" in content,
        )

    def _extract_timestamp(self, item: Dict) -> Optional[datetime]:
        """Extract and parse a timestamp from various possible fields."""
        for field_name in ("timestamp", "createdAt", "created_at", "time", "date"):
            value = item.get(field_name)
            if value is None:
                continue

            if isinstance(value, (int, float)):
                try:
                    if value > 1e12:
                        value = value / 1000.0
                    return datetime.fromtimestamp(value, tz=timezone.utc)
                except (OSError, ValueError, OverflowError):
                    continue

            if isinstance(value, str):
                parsed = self._parse_timestamp_string(value)
                if parsed:
                    return parsed

        return None

    def _parse_timestamp_string(self, value: str) -> Optional[datetime]:
        """Parse a timestamp string."""
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt
        except ValueError:
            pass

        for fmt in (
            "%Y-%m-%dT%H:%M:%S.%fZ",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S.%f",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
        ):
            try:
                dt = datetime.strptime(value, fmt)
                return dt.replace(tzinfo=timezone.utc)
            except ValueError:
                continue
        return None

    def _extract_token_count(self, item: Dict, direction: str) -> int:
        """Extract token count from various possible field names."""
        if direction == "input":
            keys = ["input_tokens", "inputTokens", "promptTokens", "prompt_tokens"]
        else:
            keys = ["output_tokens", "outputTokens", "completionTokens", "completion_tokens"]

        for key in keys:
            value = item.get(key)
            if isinstance(value, (int, float)):
                return int(value)

        usage = item.get("usage", {})
        if isinstance(usage, dict):
            for key in keys:
                value = usage.get(key)
                if isinstance(value, (int, float)):
                    return int(value)

        return 0

    def _extract_file_refs(self, text: str) -> List[str]:
        """Extract file path references from text."""
        if not text:
            return []
        pattern = r'[\w./\\-]+\.\w{1,10}'
        return list(set(re.findall(pattern, text)))[:20]

    def _has_error(self, text: str) -> bool:
        """Check if text contains error context."""
        if not text:
            return False
        return bool(re.search(
            r"(error|exception|traceback|stack trace|failed|errno)",
            text, re.IGNORECASE,
        ))

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
            (".cursorrules", "cursorrules"),
            (".cursor/rules.md", "cursor_rules_md"),
            (".cursor/mcp.json", "cursor_mcp"),
            (".cursor/hooks.json", "cursor_hooks"),
            (".cursorignore", "cursorignore"),
        ]
        for rel_path, file_type in project_files:
            full_path = workspace / rel_path
            rule_files.append(self._read_rule_file(full_path, file_type))

        # Modern .cursor/rules/*.mdc files
        rules_dir = workspace / ".cursor" / "rules"
        if rules_dir.exists():
            for mdc_file in sorted(rules_dir.glob("*.mdc")):
                rule_files.append(self._read_rule_file(mdc_file, "cursor_mdc"))

        # Custom slash commands
        commands_dir = workspace / ".cursor" / "commands"
        if commands_dir.exists():
            for cmd_file in sorted(commands_dir.glob("*.md")):
                rule_files.append(self._read_rule_file(cmd_file, "cursor_command"))

        # User-level MCP config
        user_mcp = Path.home() / ".cursor" / "mcp.json"
        rule_files.append(self._read_rule_file(user_mcp, "cursor_user_mcp"))

        # User-level hooks
        user_hooks = Path.home() / ".cursor" / "hooks.json"
        rule_files.append(self._read_rule_file(user_hooks, "cursor_user_hooks"))

        return rule_files

    def _read_rule_file(self, path: Path, file_type: str) -> RuleFileInfo:
        """Read and analyze a single rule/config file."""
        info = RuleFileInfo(
            file_path=str(path),
            file_type=file_type,
            tool=ToolType.CURSOR,
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

            if file_type in ("cursorrules", "cursor_rules_md", "cursor_mdc", "cursor_command"):
                self._analyze_markdown_file(info, content)
            elif file_type in ("cursor_mcp", "cursor_hooks", "cursor_user_mcp", "cursor_user_hooks"):
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
        """Analyze a markdown/rules file for quality signals."""
        lines = content.split("\n")

        # .mdc files can have YAML frontmatter with description, globs, alwaysApply
        if info.file_type == "cursor_mdc" and lines and lines[0].strip() == "---":
            # Skip frontmatter for section analysis
            end_idx = None
            for i, line in enumerate(lines[1:], 1):
                if line.strip() == "---":
                    end_idx = i
                    break
            if end_idx:
                lines = lines[end_idx + 1:]

        sections = [line.lstrip("#").strip() for line in lines if line.startswith("#")]
        info.sections = sections
        info.section_count = len(sections)

        lower = content.lower()
        info.has_examples = any(
            kw in lower for kw in ["example", "```", "e.g.", "for instance"]
        )
        info.has_constraints = any(
            kw in lower for kw in [
                "never", "always", "must", "don't", "do not",
                "avoid", "prefer", "required", "forbidden",
            ]
        )
        info.has_project_context = any(
            kw in lower for kw in [
                "project", "architecture", "stack", "framework",
                "directory", "structure", "overview",
            ]
        )
        info.has_style_guide = any(
            kw in lower for kw in [
                "style", "naming", "convention", "format",
                "lint", "indent", "camelcase", "snake_case",
            ]
        )
