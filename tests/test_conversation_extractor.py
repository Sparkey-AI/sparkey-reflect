"""Tests for the Conversation Extractor."""

import pytest

from sparkey_reflect.core.models import ConversationTurn, Session, SessionType, ToolType
from sparkey_reflect.insights.conversation_extractor import ConversationExtractor


@pytest.fixture
def extractor():
    return ConversationExtractor()


class TestConversationExtractor:
    def test_extract_empty(self, extractor):
        assert extractor.extract([]) == []

    def test_extract_basic_session(self, extractor, make_session, make_turn):
        session = make_session(turns=[
            make_turn(role="user", content="Fix the bug"),
            make_turn(role="assistant", content="I'll fix it now"),
        ])
        extracted = extractor.extract([session])
        assert len(extracted) == 1
        assert len(extracted[0].turns) == 2
        assert extracted[0].turns[0].role == "user"

    def test_excludes_reflect_sessions(self, extractor, make_session, make_turn):
        session = make_session(turns=[
            make_turn(content="You are Sparkey Reflect, analyze my sessions"),
            make_turn(role="assistant", content="Analyzing..."),
        ])
        extracted = extractor.extract([session])
        # Reflect session should be excluded (no turns)
        assert len(extracted) == 0

    def test_excludes_reflect_cli_sessions(self, extractor, make_session, make_turn):
        session = make_session(turns=[
            make_turn(content="Run sparkey_reflect analyze --days 7"),
        ])
        extracted = extractor.extract([session])
        assert len(extracted) == 0

    def test_tool_result_turns_summarized(self, extractor, make_session, make_turn):
        session = make_session(turns=[
            make_turn(role="user", content="Read the file"),
            make_turn(role="tool_result", content="huge file content here...", tool_name="Read"),
        ])
        extracted = extractor.extract([session])
        tool_turn = extracted[0].turns[1]
        assert tool_turn.role == "tool_result"
        assert "[tool_result: Read]" in tool_turn.content

    def test_empty_turns_with_tool_calls(self, extractor, make_session, make_turn):
        session = make_session(turns=[
            make_turn(
                role="assistant",
                content="",
                tool_calls=[{"name": "Read"}, {"name": "Edit"}],
            ),
        ])
        extracted = extractor.extract([session])
        assert len(extracted) == 1
        turn = extracted[0].turns[0]
        assert turn.content == "[tool calls only]"
        assert turn.tool_calls == ["Read", "Edit"]

    def test_empty_content_no_tools_skipped(self, extractor, make_session, make_turn):
        session = make_session(turns=[
            make_turn(role="assistant", content=""),
            make_turn(role="user", content="Hello"),
        ])
        extracted = extractor.extract([session])
        assert len(extracted[0].turns) == 1


class TestContentCleaning:
    def test_strip_large_code_blocks(self, extractor):
        content = "Look at this:\n```python\n" + "x = 1\n" * 20 + "```"
        cleaned = extractor._clean_content(content, "user")
        assert "[code: python," in cleaned
        assert "lines]" in cleaned

    def test_keep_small_code_blocks(self, extractor):
        content = "Here:\n```python\nx = 1\ny = 2\n```"
        cleaned = extractor._clean_content(content, "user")
        assert "x = 1" in cleaned

    def test_strip_base64(self, extractor):
        b64 = "A" * 200 + "=="
        content = f"Image data: {b64}"
        cleaned = extractor._clean_content(content, "user")
        assert "[binary content omitted]" in cleaned
        assert "A" * 200 not in cleaned

    def test_strip_diffs(self, extractor):
        diff = (
            "Here's the change:\n"
            "--- a/file.py\n"
            "+++ b/file.py\n"
            "@@ -1,3 +1,3 @@\n"
            "-old line\n"
            "+new line\n"
            " context\n"
            "That's it."
        )
        cleaned = extractor._clean_content(diff, "assistant")
        assert "[diff:" in cleaned

    def test_collapse_multiple_blank_lines(self, extractor):
        content = "first\n\n\n\n\nsecond"
        cleaned = extractor._clean_content(content, "user")
        assert "\n\n\n" not in cleaned


class TestTurnTruncation:
    def test_user_turns_use_larger_limit(self):
        extractor = ConversationExtractor(max_user_turn_chars=100, max_assistant_turn_chars=50)
        # Use words instead of repeated chars to avoid base64 pattern match
        user_turn = ConversationTurn(role="user", content="fix the bug " * 30)
        result = extractor._extract_turn(user_turn)
        assert len(result.content) < len("fix the bug " * 30)
        assert "truncated" in result.content

    def test_assistant_turns_use_smaller_limit(self):
        extractor = ConversationExtractor(max_user_turn_chars=100, max_assistant_turn_chars=50)
        asst_turn = ConversationTurn(role="assistant", content="x" * 200)
        result = extractor._extract_turn(asst_turn)
        assert len(result.content) < 100


class TestTokenEstimation:
    def test_empty(self, extractor):
        assert extractor.estimate_tokens([]) == 0

    def test_basic_estimation(self, extractor, make_session, make_turn):
        session = make_session(turns=[
            make_turn(content="Hello world test prompt"),
        ])
        extracted = extractor.extract([session])
        tokens = extractor.estimate_tokens(extracted)
        assert tokens > 0


class TestPromptFormatting:
    def test_to_prompt_text(self, extractor, make_session, make_turn):
        session = make_session(
            session_id="abc123def456",
            turns=[
                make_turn(content="Fix the bug"),
                make_turn(role="assistant", content="Fixed it"),
            ],
        )
        extracted = extractor.extract([session])
        text = extractor.to_prompt_text(extracted)
        assert "### Session:" in text
        assert "USER" in text
        assert "ASSISTANT" in text
        assert "Fix the bug" in text
