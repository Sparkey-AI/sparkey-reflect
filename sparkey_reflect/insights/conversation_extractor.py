"""
Conversation Extractor

Extracts a clean, token-efficient representation of conversations for the LLM.
Strips large code blocks, tool results, diffs, and binary content while
preserving user prompts, assistant reasoning, tool call names, and error messages.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from sparkey_reflect.core.models import Session


# Patterns for detecting content that should be stripped
DIFF_PATTERN = re.compile(r'^[\+\-]{3}\s|^@@\s', re.MULTILINE)
BASE64_PATTERN = re.compile(r'[A-Za-z0-9+/]{100,}={0,2}')
FILE_LISTING_PATTERN = re.compile(
    r'(?:^[ \t]*(?:[-d][-rwx]{9}|total \d+).*\n?){5,}', re.MULTILINE
)

# Patterns for detecting reflect analysis sessions (should be excluded)
REFLECT_SESSION_MARKERS = [
    "You are Sparkey Reflect",
    "sparkey_reflect analyze",
]


@dataclass
class ExtractedTurn:
    """A cleaned conversation turn."""
    role: str
    content: str
    tool_calls: List[str] = field(default_factory=list)


@dataclass
class ExtractedSession:
    """A cleaned session summary for LLM consumption."""
    session_id: str
    timestamp: Optional[str] = None
    workspace: Optional[str] = None
    session_type: Optional[str] = None
    duration_minutes: float = 0.0
    turns: List[ExtractedTurn] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            "workspace": self.workspace,
            "session_type": self.session_type,
            "duration_minutes": self.duration_minutes,
            "turns": [
                {"role": t.role, "content": t.content, "tool_calls": t.tool_calls}
                for t in self.turns
            ],
        }


class ConversationExtractor:
    """Extracts clean, token-efficient conversation data from sessions."""

    def __init__(
        self,
        max_code_block_lines: int = 5,
        max_user_turn_chars: int = 4000,
        max_assistant_turn_chars: int = 800,
        max_turns_per_session: int = 40,
    ):
        self.max_code_block_lines = max_code_block_lines
        self.max_user_turn_chars = max_user_turn_chars
        self.max_assistant_turn_chars = max_assistant_turn_chars
        self.max_turns_per_session = max_turns_per_session

    def extract(self, sessions: List[Session]) -> List[ExtractedSession]:
        """Extract cleaned conversation data from all sessions."""
        extracted = []
        for session in sessions:
            es = self._extract_session(session)
            if es.turns:
                extracted.append(es)
        return extracted

    def estimate_tokens(self, extracted: List[ExtractedSession]) -> int:
        """Estimate token count using ~4 chars per token heuristic."""
        total_chars = 0
        for es in extracted:
            total_chars += len(es.session_id or "")
            total_chars += len(es.workspace or "")
            for turn in es.turns:
                total_chars += len(turn.content)
                total_chars += sum(len(tc) for tc in turn.tool_calls)
        return total_chars // 4

    def to_prompt_text(self, extracted: List[ExtractedSession]) -> str:
        """Format extracted sessions as readable text for the LLM prompt."""
        parts = []
        for es in extracted:
            header = f"### Session: {es.session_id[:12]}"
            if es.timestamp:
                header += f" | {es.timestamp}"
            if es.workspace:
                # Show just the last path component
                ws_short = es.workspace.rstrip("/").rsplit("/", 1)[-1]
                header += f" | {ws_short}"
            if es.session_type:
                header += f" | {es.session_type}"
            if es.duration_minutes:
                header += f" | {es.duration_minutes:.0f}min"
            parts.append(header)

            for turn in es.turns:
                role_label = turn.role.upper()
                parts.append(f"**{role_label}**: {turn.content}")
                if turn.tool_calls:
                    parts.append(f"  Tools: {', '.join(turn.tool_calls)}")

            parts.append("")  # blank line between sessions
        return "\n".join(parts)

    # =========================================================================
    # Internal
    # =========================================================================

    def _is_reflect_session(self, session: Session) -> bool:
        """Detect sessions that are reflect analysis runs (should be excluded)."""
        for turn in session.turns[:5]:  # check only the first few turns
            content = turn.content or ""
            for marker in REFLECT_SESSION_MARKERS:
                if marker in content:
                    return True
        return False

    def _extract_session(self, session: Session) -> ExtractedSession:
        """Extract a single session into cleaned form."""
        if self._is_reflect_session(session):
            return ExtractedSession(session_id=session.session_id)

        es = ExtractedSession(
            session_id=session.session_id,
            timestamp=session.start_time.isoformat() if session.start_time else None,
            workspace=session.workspace_path,
            session_type=session.session_type.value if session.session_type else None,
            duration_minutes=session.duration_minutes,
        )

        # Sample turns: keep first few, last few, and evenly spaced middle
        turns = session.turns
        if len(turns) > self.max_turns_per_session:
            keep_start = 10
            keep_end = 10
            middle = turns[keep_start:-keep_end] if keep_end else turns[keep_start:]
            step = max(1, len(middle) // (self.max_turns_per_session - keep_start - keep_end))
            sampled_middle = middle[::step]
            turns = list(turns[:keep_start]) + list(sampled_middle) + list(turns[-keep_end:])

        for turn in turns:
            extracted_turn = self._extract_turn(turn)
            if extracted_turn:
                es.turns.append(extracted_turn)

        return es

    def _extract_turn(self, turn) -> Optional[ExtractedTurn]:
        """Extract and clean a single turn."""
        # Skip tool result turns — replace with summary
        if turn.role == "tool_result":
            tool_name = turn.tool_name or "unknown"
            return ExtractedTurn(
                role="tool_result",
                content=f"[tool_result: {tool_name}]",
            )

        content = turn.content or ""
        if not content.strip():
            # Still capture tool calls even with empty content
            if turn.tool_calls:
                tool_names = [tc.get("name", "unknown") for tc in turn.tool_calls]
                return ExtractedTurn(
                    role=turn.role,
                    content="[tool calls only]",
                    tool_calls=tool_names,
                )
            return None

        cleaned = self._clean_content(content, turn.role)

        # Truncate long turns — preserve user prompts (key coaching signal)
        max_chars = (
            self.max_user_turn_chars if turn.role == "user"
            else self.max_assistant_turn_chars
        )
        if len(cleaned) > max_chars:
            cleaned = cleaned[:max_chars] + "... [truncated]"

        tool_names = []
        if turn.tool_calls:
            tool_names = [tc.get("name", "unknown") for tc in turn.tool_calls]

        return ExtractedTurn(
            role=turn.role,
            content=cleaned,
            tool_calls=tool_names,
        )

    def _clean_content(self, content: str, role: str) -> str:
        """Clean content by stripping large code blocks, diffs, binary data."""
        # Strip large code blocks
        content = self._strip_code_blocks(content)

        # Strip diffs/patches
        content = self._strip_diffs(content)

        # Strip base64/binary content
        content = BASE64_PATTERN.sub("[binary content omitted]", content)

        # Strip large file listings
        content = FILE_LISTING_PATTERN.sub(
            lambda m: f"[file listing: {m.group().count(chr(10)) + 1} entries]",
            content,
        )

        # Collapse multiple blank lines
        content = re.sub(r'\n{3,}', '\n\n', content)

        return content.strip()

    def _strip_code_blocks(self, content: str) -> str:
        """Replace large code blocks with summaries."""
        def replace_block(match):
            lang = match.group(1) or ""
            code = match.group(2)
            line_count = code.count("\n") + 1
            if line_count <= self.max_code_block_lines:
                return match.group(0)  # keep small blocks
            lang_label = lang.strip() if lang.strip() else "code"
            return f"[code: {lang_label}, {line_count} lines]"

        return re.sub(
            r'```(\w*)\n(.*?)```',
            replace_block,
            content,
            flags=re.DOTALL,
        )

    def _strip_diffs(self, content: str) -> str:
        """Replace inline diffs/patches with summaries."""
        if not DIFF_PATTERN.search(content):
            return content

        lines = content.split("\n")
        result = []
        in_diff = False
        diff_file = ""
        added = 0
        removed = 0

        for line in lines:
            if line.startswith("--- ") or line.startswith("+++ "):
                if not in_diff:
                    in_diff = True
                    added = 0
                    removed = 0
                if line.startswith("+++ "):
                    diff_file = line[4:].strip()
                continue

            if in_diff:
                if line.startswith("@@"):
                    continue
                if line.startswith("+"):
                    added += 1
                    continue
                if line.startswith("-"):
                    removed += 1
                    continue
                # End of diff hunk
                if diff_file or added or removed:
                    file_label = diff_file or "file"
                    result.append(
                        f"[diff: {file_label}, +{added}/-{removed} lines]"
                    )
                in_diff = False
                diff_file = ""
                added = 0
                removed = 0
                result.append(line)
            else:
                result.append(line)

        # Flush remaining diff
        if in_diff and (diff_file or added or removed):
            file_label = diff_file or "file"
            result.append(f"[diff: {file_label}, +{added}/-{removed} lines]")

        return "\n".join(result)
