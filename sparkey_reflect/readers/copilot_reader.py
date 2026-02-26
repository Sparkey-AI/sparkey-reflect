"""
Copilot Reader

Reads GitHub Copilot data from two sources:

1. Reflect's own trace files (primary):
   ~/.sparkey/reflect/copilot_traces/<session>.json
   Captured by the Reflect for Copilot extension via chat participant hook.

2. VS Code log files (fallback):
   ~/Library/Application Support/Code/logs/<date>/GitHub Copilot*/*.log
   Provides completion metrics when the extension's capture hook isn't active.

Also reads rule files:
   .github/copilot-instructions.md, .github/instructions/*.instructions.md,
   AGENTS.md, .agent.md, .github/prompts/*.md, etc.
"""

import hashlib
import json
import logging
import os
import platform
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from sparkey_reflect.core.models import (
    CompletionEvent,
    ConversationTurn,
    RuleFileInfo,
    Session,
    SessionType,
    ToolType,
)
from sparkey_reflect.readers.base_reader import BaseReader

logger = logging.getLogger(__name__)

# Platform-specific VS Code log directories
if platform.system() == "Darwin":
    VSCODE_LOGS_DIR = Path.home() / "Library" / "Application Support" / "Code" / "logs"
elif platform.system() == "Windows":
    _appdata = os.environ.get("APPDATA", "")
    VSCODE_LOGS_DIR = Path(_appdata) / "Code" / "logs" if _appdata else Path.home() / "Code" / "logs"
else:
    VSCODE_LOGS_DIR = Path.home() / ".config" / "Code" / "logs"

# Reflect's own Copilot trace directory
COPILOT_TRACES_DIR = Path.home() / ".sparkey" / "reflect" / "copilot_traces"

# Log line parsing
LOG_LINE_RE = re.compile(r"(?P<timestamp>\S+)\s+(?P<level>\S+)\s+(?P<message>.*)")

# Event type classification patterns
EVENT_PATTERNS = {
    "suggestion_generated": re.compile(
        r"(?:suggestion|completion)\s+(?:generated|shown)", re.IGNORECASE,
    ),
    "completion_accepted": re.compile(
        r"(?:completion|suggestion)\s+accepted", re.IGNORECASE,
    ),
    "completion_rejected": re.compile(
        r"(?:completion|suggestion)\s+(?:rejected|dismissed|discarded)", re.IGNORECASE,
    ),
    "chat_response": re.compile(
        r"chat\s+(?:response|reply|answer)", re.IGNORECASE,
    ),
}

# Field extraction regexes
MODEL_RE = re.compile(r"model[\"']?\s*[:=]\s*[\"']?(?P<model>[\w\-./:]+)")
INPUT_TOKENS_RE = re.compile(r"input[_\s]?tokens?[\"']?\s*[:=]\s*(?P<input_tokens>\d+)")
OUTPUT_TOKENS_RE = re.compile(r"output[_\s]?tokens?[\"']?\s*[:=]\s*(?P<output_tokens>\d+)")
FILE_PATH_RE = re.compile(r"(?:file|path)[\"']?\s*[:=]\s*[\"']?(?P<file_path>[^\s\"',}]+)")
LANGUAGE_RE = re.compile(r"(?:language|languageId)[\"']?\s*[:=]\s*[\"']?(?P<language>\w+)")
LINES_SUGGESTED_RE = re.compile(r"(?:lines?[_\s]?suggested|numLines)[\"']?\s*[:=]\s*(?P<lines>\d+)")

# Session type classification patterns
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


class CopilotReader(BaseReader):
    """Reader for GitHub Copilot data (trace files + VS Code logs)."""

    def get_tool_type(self) -> ToolType:
        return ToolType.COPILOT

    def is_available(self) -> bool:
        # Check Reflect's own trace files
        if COPILOT_TRACES_DIR.exists() and any(COPILOT_TRACES_DIR.glob("*.json")):
            return True
        # Check VS Code log files
        if VSCODE_LOGS_DIR.exists():
            return bool(self._find_log_files())
        return False

    def get_data_locations(self) -> List[str]:
        locations = []
        if COPILOT_TRACES_DIR.exists():
            locations.append(str(COPILOT_TRACES_DIR))
        if VSCODE_LOGS_DIR.exists():
            locations.append(str(VSCODE_LOGS_DIR))
        return locations

    def get_history_range(self) -> Optional[Tuple[datetime, datetime]]:
        earliest = None
        latest = None

        # Check trace files
        if COPILOT_TRACES_DIR.exists():
            for trace_file in COPILOT_TRACES_DIR.glob("*.json"):
                self._update_range_from_file(trace_file, earliest, latest)

        # Check log files
        for log_file in self._find_log_files():
            try:
                mtime = datetime.fromtimestamp(
                    log_file.stat().st_mtime, tz=timezone.utc
                )
                ctime = datetime.fromtimestamp(
                    log_file.stat().st_ctime, tz=timezone.utc
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

    def _update_range_from_file(self, path: Path, earliest, latest):
        """Update earliest/latest from a file's timestamps."""
        try:
            mtime = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
            ctime = datetime.fromtimestamp(path.stat().st_ctime, tz=timezone.utc)
            ts = min(mtime, ctime)
            if earliest is None or ts < earliest:
                earliest = ts
            if latest is None or mtime > latest:
                latest = mtime
        except OSError:
            pass

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

        # 1. Read from Reflect's trace files (primary - full conversations)
        trace_sessions = self._read_trace_sessions(since, until, workspace_path)
        sessions.extend(trace_sessions)

        # 2. Read from VS Code logs (fallback - completion events grouped into sessions)
        log_sessions = self._read_log_sessions(since, until)
        sessions.extend(log_sessions)

        sessions.sort(key=lambda s: s.start_time or datetime.min.replace(tzinfo=timezone.utc))
        return sessions

    def _read_trace_sessions(
        self,
        since: Optional[datetime],
        until: Optional[datetime],
        workspace_path: Optional[str],
    ) -> List[Session]:
        """Read full conversation sessions from Reflect's trace files."""
        if not COPILOT_TRACES_DIR.exists():
            return []

        sessions = []
        for trace_file in COPILOT_TRACES_DIR.glob("*.json"):
            # Quick mtime filter
            if since:
                try:
                    mtime = datetime.fromtimestamp(
                        trace_file.stat().st_mtime, tz=timezone.utc
                    )
                    since_aware = since if since.tzinfo else since.replace(tzinfo=timezone.utc)
                    if mtime < since_aware:
                        continue
                except OSError:
                    continue

            try:
                session = self._parse_trace_file(trace_file)
                if session is None:
                    continue

                # Apply time filters
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

                # Workspace filter
                if workspace_path and session.workspace_path:
                    if workspace_path not in session.workspace_path:
                        continue

                sessions.append(session)
            except Exception as e:
                logger.warning("Error reading trace file %s: %s", trace_file, e)

        return sessions

    def _parse_trace_file(self, path: Path) -> Optional[Session]:
        """Parse a Reflect Copilot trace JSON file into a Session."""
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            logger.debug("Cannot parse %s: %s", path, e)
            return None

        if not isinstance(data, dict):
            return None

        session_id = data.get("sessionId") or data.get("id") or path.stem
        model = data.get("model")
        workspace = data.get("workspace") or data.get("cwd")

        turns = []
        total_input = 0
        total_output = 0

        # Parse conversation turns
        raw_turns = data.get("turns") or data.get("messages") or data.get("conversation") or []
        for raw in raw_turns:
            if not isinstance(raw, dict):
                continue

            role = raw.get("role", "")
            if role not in ("user", "assistant", "system", "tool"):
                continue

            content = raw.get("content") or raw.get("text") or ""
            if isinstance(content, list):
                content = "\n".join(
                    b.get("text", "") if isinstance(b, dict) else str(b) for b in content
                )

            tool_calls = []
            for tc in raw.get("toolCalls", raw.get("tool_calls", [])):
                if isinstance(tc, dict):
                    tool_calls.append({
                        "name": tc.get("name") or tc.get("function", {}).get("name", ""),
                        "id": tc.get("id", ""),
                    })

            timestamp = self._parse_timestamp(
                raw.get("timestamp") or raw.get("createdAt")
            )

            in_tokens = raw.get("input_tokens") or raw.get("inputTokens") or 0
            out_tokens = raw.get("output_tokens") or raw.get("outputTokens") or 0
            total_input += in_tokens
            total_output += out_tokens

            if not model and raw.get("model"):
                model = raw["model"]

            turns.append(ConversationTurn(
                role="tool_result" if role == "tool" else role,
                content=content,
                timestamp=timestamp,
                tool_calls=tool_calls,
                input_tokens=in_tokens,
                output_tokens=out_tokens,
                file_references=self._extract_file_refs(content) if role == "user" else [],
                has_error_context=self._has_error(content) if role == "user" else False,
                has_code_snippet="```" in content,
            ))

        if not turns:
            return None

        timestamps = [t.timestamp for t in turns if t.timestamp]
        start_time = min(timestamps) if timestamps else self._parse_timestamp(data.get("timestamp"))
        end_time = max(timestamps) if timestamps else start_time

        duration = 0.0
        if start_time and end_time:
            duration = (end_time - start_time).total_seconds() / 60.0

        return Session(
            session_id=f"cop_{session_id}",
            tool=ToolType.COPILOT,
            turns=turns,
            start_time=start_time,
            end_time=end_time,
            duration_minutes=max(0, duration),
            workspace_path=workspace,
            model=model,
            total_input_tokens=total_input,
            total_output_tokens=total_output,
            session_type=self._classify_session(turns),
            metadata={
                "source": "trace_file",
                "file_path": str(path),
            },
        )

    # =========================================================================
    # VS Code Log Parsing (fallback)
    # =========================================================================

    def _read_log_sessions(
        self,
        since: Optional[datetime],
        until: Optional[datetime],
    ) -> List[Session]:
        """Parse VS Code Copilot logs into pseudo-sessions grouped by time windows."""
        events = self._parse_all_log_files(since)
        if not events:
            return []

        # Apply until filter
        if until:
            until_aware = until if until.tzinfo else until.replace(tzinfo=timezone.utc)
            events = [e for e in events if e.timestamp <= until_aware]

        # Group events into 30-minute windows (pseudo-sessions)
        return self._group_events_into_sessions(events)

    def _parse_all_log_files(self, since: Optional[datetime]) -> List[CompletionEvent]:
        """Parse all Copilot log files into CompletionEvent objects."""
        events = []
        for log_file in self._find_log_files():
            if since:
                try:
                    mtime = datetime.fromtimestamp(
                        log_file.stat().st_mtime, tz=timezone.utc
                    )
                    since_aware = since if since.tzinfo else since.replace(tzinfo=timezone.utc)
                    if mtime < since_aware:
                        continue
                except OSError:
                    continue

            try:
                file_events = self._parse_log_file(log_file)
                if since:
                    since_aware = since if since.tzinfo else since.replace(tzinfo=timezone.utc)
                    file_events = [e for e in file_events if e.timestamp >= since_aware]
                events.extend(file_events)
            except Exception as e:
                logger.warning("Error parsing log file %s: %s", log_file, e)

        events.sort(key=lambda e: e.timestamp)
        return events

    def _parse_log_file(self, path: Path) -> List[CompletionEvent]:
        """Parse a single VS Code Copilot log file."""
        events = []
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.rstrip("\n\r")
                    if not line:
                        continue

                    match = LOG_LINE_RE.match(line)
                    if not match:
                        continue

                    ts_str = match.group("timestamp")
                    message = match.group("message")

                    event = self._parse_log_event(ts_str, message, str(path))
                    if event:
                        events.append(event)
        except OSError as e:
            logger.debug("Cannot read %s: %s", path, e)

        return events

    def _parse_log_event(
        self, ts_str: str, message: str, source_file: str,
    ) -> Optional[CompletionEvent]:
        """Parse a single log line into a CompletionEvent."""
        event_type = None
        for etype, pattern in EVENT_PATTERNS.items():
            if pattern.search(message):
                event_type = etype
                break

        if not event_type:
            return None

        timestamp = self._parse_timestamp(ts_str)
        if not timestamp:
            return None

        # Extract fields
        model = None
        model_match = MODEL_RE.search(message)
        if model_match:
            model = model_match.group("model")

        language = "unknown"
        lang_match = LANGUAGE_RE.search(message)
        if lang_match:
            language = lang_match.group("language")

        file_path = None
        fp_match = FILE_PATH_RE.search(message)
        if fp_match:
            file_path = fp_match.group("file_path")

        lines_suggested = 0
        lines_match = LINES_SUGGESTED_RE.search(message)
        if lines_match:
            lines_suggested = int(lines_match.group("lines"))

        accepted = event_type == "completion_accepted"

        # Generate unique event ID
        event_id = hashlib.md5(
            f"{ts_str}:{event_type}:{message[:50]}".encode()
        ).hexdigest()[:12]

        return CompletionEvent(
            event_id=event_id,
            timestamp=timestamp,
            language=language,
            suggestion_length=lines_suggested,
            accepted=accepted,
            model=model,
            file_path=file_path,
            metadata={
                "event_type": event_type,
                "source_file": source_file,
            },
        )

    def _group_events_into_sessions(
        self, events: List[CompletionEvent],
    ) -> List[Session]:
        """Group completion events into 30-minute pseudo-sessions."""
        if not events:
            return []

        sessions = []
        window_minutes = 30

        current_events: List[CompletionEvent] = [events[0]]
        for event in events[1:]:
            last_ts = current_events[-1].timestamp
            gap = (event.timestamp - last_ts).total_seconds() / 60.0

            if gap > window_minutes:
                # Close current window, start new one
                session = self._events_to_session(current_events)
                if session:
                    sessions.append(session)
                current_events = [event]
            else:
                current_events.append(event)

        # Handle last window
        if current_events:
            session = self._events_to_session(current_events)
            if session:
                sessions.append(session)

        return sessions

    def _events_to_session(self, events: List[CompletionEvent]) -> Optional[Session]:
        """Convert a group of CompletionEvents into a pseudo-Session."""
        if not events:
            return None

        start_time = events[0].timestamp
        end_time = events[-1].timestamp
        duration = (end_time - start_time).total_seconds() / 60.0

        # Build turns representing accepted/rejected completions
        turns = []
        for event in events:
            action = "accepted" if event.accepted else "suggested"
            content = f"Copilot {action} completion in {event.language}"
            if event.file_path:
                content += f" ({event.file_path})"

            turns.append(ConversationTurn(
                role="assistant",
                content=content,
                timestamp=event.timestamp,
                file_references=[event.file_path] if event.file_path else [],
            ))

        # Session ID from timestamp
        session_id = f"cop_log_{start_time.strftime('%Y%m%d_%H%M%S')}"

        # Compute acceptance stats for metadata
        total = len(events)
        accepted = sum(1 for e in events if e.accepted)
        languages = list(set(e.language for e in events if e.language != "unknown"))

        return Session(
            session_id=session_id,
            tool=ToolType.COPILOT,
            turns=turns,
            start_time=start_time,
            end_time=end_time,
            duration_minutes=max(0, duration),
            model=events[0].model,
            session_type=SessionType.CODING,
            metadata={
                "source": "vscode_logs",
                "completion_events": total,
                "completions_accepted": accepted,
                "acceptance_rate": accepted / total if total > 0 else 0,
                "languages": languages,
                "events": [
                    {
                        "event_id": e.event_id,
                        "timestamp": e.timestamp.isoformat(),
                        "language": e.language,
                        "accepted": e.accepted,
                        "suggestion_length": e.suggestion_length,
                        "model": e.model,
                        "file_path": e.file_path,
                    }
                    for e in events
                ],
            },
        )

    def _find_log_files(self) -> List[Path]:
        """Find all Copilot log files under the VS Code logs directory."""
        if not VSCODE_LOGS_DIR.exists():
            return []

        files = []
        for pattern in [
            "*/GitHub Copilot*/**/*.log",
            "*/GitHub Copilot*/*.log",
        ]:
            files.extend(VSCODE_LOGS_DIR.glob(pattern))

        # Deduplicate and sort by modification time (newest first)
        seen = set()
        unique = []
        for f in files:
            if f not in seen:
                seen.add(f)
                unique.append(f)
        unique.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        return unique

    # =========================================================================
    # Helpers
    # =========================================================================

    def _parse_timestamp(self, value) -> Optional[datetime]:
        """Parse various timestamp formats."""
        if value is None:
            return None

        if isinstance(value, (int, float)):
            try:
                if value > 1e12:
                    value = value / 1000
                return datetime.fromtimestamp(value, tz=timezone.utc)
            except (ValueError, OSError):
                return None

        if isinstance(value, str):
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
                "%Y-%m-%d %H:%M:%S.%f",
                "%Y-%m-%d %H:%M:%S",
            ):
                try:
                    dt = datetime.strptime(value, fmt)
                    return dt.replace(tzinfo=timezone.utc)
                except ValueError:
                    continue

        return None

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
        """Classify session type from conversation content."""
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

        # Primary instruction file
        project_files = [
            (".github/copilot-instructions.md", "copilot_instructions"),
            ("AGENTS.md", "agents_md"),
            (".agent.md", "agent_md"),
            (".vscode/settings.json", "vscode_settings"),
        ]
        for rel_path, file_type in project_files:
            full_path = workspace / rel_path
            rule_files.append(self._read_rule_file(full_path, file_type))

        # Path-scoped instruction files
        instructions_dir = workspace / ".github" / "instructions"
        if instructions_dir.exists():
            for inst_file in sorted(instructions_dir.glob("*.instructions.md")):
                rule_files.append(self._read_rule_file(inst_file, "copilot_scoped_instructions"))

        # Prompt files / slash commands
        prompts_dir = workspace / ".github" / "prompts"
        if prompts_dir.exists():
            for prompt_file in sorted(prompts_dir.glob("*.md")):
                rule_files.append(self._read_rule_file(prompt_file, "copilot_prompt"))

        # Copilot skills
        skills_file = workspace / ".github" / "copilot-skills.md"
        rule_files.append(self._read_rule_file(skills_file, "copilot_skills"))

        # Nested AGENTS.md (one level deep)
        try:
            for subdir in workspace.iterdir():
                if subdir.is_dir() and not subdir.name.startswith("."):
                    agents_md = subdir / "AGENTS.md"
                    if agents_md.exists():
                        rule_files.append(
                            self._read_rule_file(agents_md, "agents_md_nested")
                        )
        except OSError:
            pass

        return rule_files

    def _read_rule_file(self, path: Path, file_type: str) -> RuleFileInfo:
        """Read and analyze a single rule/config file."""
        info = RuleFileInfo(
            file_path=str(path),
            file_type=file_type,
            tool=ToolType.COPILOT,
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

            if file_type in ("copilot_instructions", "copilot_scoped_instructions",
                             "agents_md", "agent_md", "agents_md_nested",
                             "copilot_prompt", "copilot_skills"):
                self._analyze_markdown_file(info, content)
            elif file_type == "vscode_settings":
                info.has_tool_config = True
                try:
                    data = json.loads(content)
                    # Count Copilot-related settings
                    copilot_keys = [k for k in data if "copilot" in k.lower() or "github" in k.lower()]
                    info.section_count = len(copilot_keys)
                except json.JSONDecodeError:
                    pass
        except OSError as e:
            logger.debug("Cannot read %s: %s", path, e)
            info.exists = False

        return info

    def _analyze_markdown_file(self, info: RuleFileInfo, content: str):
        """Analyze a markdown instruction file for quality signals."""
        lines = content.split("\n")

        # Handle applyTo: header in scoped instruction files
        if info.file_type == "copilot_scoped_instructions":
            for i, line in enumerate(lines):
                if line.strip().lower().startswith("applyto:"):
                    lines = lines[i + 1:]
                    break

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
