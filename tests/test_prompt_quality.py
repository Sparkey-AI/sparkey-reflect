"""Tests for the Prompt Quality analyzer."""

import pytest

from sparkey_reflect.analyzers.prompt_quality import PromptQualityAnalyzer
from sparkey_reflect.core.models import ConversationTurn, Session, ToolType


@pytest.fixture
def analyzer():
    return PromptQualityAnalyzer()


class TestPromptQualityAnalyzer:
    def test_key_and_name(self, analyzer):
        assert analyzer.get_key() == "prompt_quality"
        assert analyzer.get_name() == "Prompt Quality"

    def test_empty_sessions(self, analyzer):
        result = analyzer.analyze([])
        assert result.score == 0
        assert result.session_count == 0

    def test_high_quality_prompt(self, analyzer, make_session, make_turn):
        session = make_session(turns=[
            make_turn(
                content="Fix the TypeError in api/auth.py line 42 where get_user returns None instead of raising NotFoundError. Keep backward compatibility with existing callers.",
                file_references=["api/auth.py"],
                has_error_context=True,
                has_code_snippet=True,
            ),
        ])
        result = analyzer.analyze([session])
        assert result.score > 50
        assert result.metrics["prompts_analyzed"] == 1

    def test_low_quality_prompt(self, analyzer, make_session, make_turn):
        session = make_session(turns=[
            make_turn(content="help me with something"),
        ])
        result = analyzer.analyze([session])
        assert result.score < 40

    def test_multiple_sessions(self, analyzer, sample_sessions):
        result = analyzer.analyze(sample_sessions)
        assert result.score > 0
        assert result.session_count == 3
        assert result.metrics["prompts_analyzed"] > 0

    def test_score_bounded_0_100(self, analyzer, make_session, make_turn):
        # Very short prompt
        session = make_session(turns=[make_turn(content="x")])
        result = analyzer.analyze([session])
        assert 0 <= result.score <= 100


class TestSpecificityScoring:
    def test_file_references_boost(self, analyzer, make_turn):
        turn_with = make_turn(content="Fix the bug in this file", file_references=["auth.py"])
        turn_without = make_turn(content="Fix the bug in this file")
        score_with = analyzer._score_specificity(turn_with)
        score_without = analyzer._score_specificity(turn_without)
        assert score_with > score_without

    def test_identifiers_boost(self, analyzer, make_turn):
        turn = make_turn(content="Rename the get_user_by_id function to fetch_user_by_id in the auth module")
        score = analyzer._score_specificity(turn)
        assert score > 5

    def test_line_numbers_boost(self, analyzer, make_turn):
        turn_with = make_turn(content="Fix the error on line 42 in the auth module")
        turn_without = make_turn(content="Fix the error in the auth module")
        assert analyzer._score_specificity(turn_with) > analyzer._score_specificity(turn_without)

    def test_vague_patterns_penalized(self, analyzer, make_turn):
        vague = make_turn(content="Can you help me with something please? I need things done somehow")
        specific = make_turn(content="Add a retry decorator to the fetch_data function in utils.py")
        assert analyzer._score_specificity(specific) > analyzer._score_specificity(vague)

    def test_very_short_prompt_low_score(self, analyzer, make_turn):
        turn = make_turn(content="fix it")
        score = analyzer._score_specificity(turn)
        assert score <= 3


class TestContextRichnessScoring:
    def test_error_context_boost(self, analyzer, make_turn):
        turn = make_turn(content="Debug the error", has_error_context=True)
        score = analyzer._score_context_richness(turn)
        assert score >= 6

    def test_code_snippet_boost(self, analyzer, make_turn):
        turn = make_turn(content="Fix this code", has_code_snippet=True)
        score = analyzer._score_context_richness(turn)
        assert score >= 5

    def test_expected_behavior_boost(self, analyzer, make_turn):
        turn = make_turn(content="This function should return a list of users")
        score = analyzer._score_context_richness(turn)
        assert score >= 4

    def test_constraints_boost(self, analyzer, make_turn):
        turn = make_turn(content="Refactor this without breaking backward compatibility")
        score = analyzer._score_context_richness(turn)
        assert score >= 3


class TestClarityScoring:
    def test_structured_prompt_boost(self, analyzer, make_turn):
        turn = make_turn(content="1. Create the model\n2. Add the endpoint\n3. Write tests")
        score = analyzer._score_clarity(turn)
        assert score > 5

    def test_question_only_penalized(self, analyzer, make_turn):
        question = make_turn(content="What is this?")
        # Statement with clear structure and scope to score higher
        statement = make_turn(
            content="Explain how the auth middleware works in this codebase. Specifically, show the request validation flow."
        )
        assert analyzer._score_clarity(statement) > analyzer._score_clarity(question)

    def test_markdown_formatting_boost(self, analyzer, make_turn):
        turn = make_turn(content="Fix the `get_user` function in **auth.py** to handle None")
        score = analyzer._score_clarity(turn)
        assert score >= 3  # markdown bonus


class TestEfficiencyScoring:
    def test_efficient_short_with_context(self, analyzer, make_turn, make_session):
        turn = make_turn(
            content="Fix the null check in auth.py",
            file_references=["auth.py"],
        )
        session = make_session(turns=[turn])
        score = analyzer._score_efficiency(turn, session)
        assert score >= 15  # base + efficiency bonus

    def test_very_long_prompt_penalized(self, analyzer, make_turn, make_session):
        turn = make_turn(content="word " * 600)
        session = make_session(turns=[turn])
        score = analyzer._score_efficiency(turn, session)
        assert score < 15

    def test_filler_words_penalized(self, analyzer, make_turn, make_session):
        turn = make_turn(
            content="Please could you thank you I was wondering if maybe you could help me"
        )
        session = make_session(turns=[turn])
        score = analyzer._score_efficiency(turn, session)
        assert score < 15
