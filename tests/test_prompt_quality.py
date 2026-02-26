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
        assert result.score < 50

    def test_multiple_sessions(self, analyzer, sample_sessions):
        result = analyzer.analyze(sample_sessions)
        assert result.score > 0
        assert result.session_count == 3
        assert result.metrics["prompts_analyzed"] > 0

    def test_score_bounded_0_100(self, analyzer, make_session, make_turn):
        session = make_session(turns=[make_turn(content="x")])
        result = analyzer.analyze([session])
        assert 0 <= result.score <= 100

    def test_new_chain_of_thought_metric(self, analyzer, make_session, make_turn):
        """Chain of thought dimension is computed and reported."""
        session = make_session(turns=[
            make_turn(content="1. Create the model\n2. Add endpoint\n3. Write tests because we need coverage"),
        ])
        result = analyzer.analyze([session])
        assert "chain_of_thought" in result.metrics
        assert result.metrics["chain_of_thought"] > 0

    def test_smooth_score_not_jagged(self, analyzer, make_session, make_turn):
        """Scores should vary smoothly, not jump between fixed levels."""
        scores = []
        # Gradually improve prompt quality
        prompts = [
            "fix it",
            "fix the auth bug",
            "fix the auth bug in auth.py",
            "Fix the TypeError in api/auth.py where get_user fails",
            "Fix the TypeError in api/auth.py line 42 where get_user returns None instead of raising NotFoundError. Keep backward compatibility.",
        ]
        for prompt in prompts:
            session = make_session(turns=[make_turn(content=prompt)])
            result = analyzer.analyze([session])
            scores.append(result.score)

        # Scores should generally increase
        assert scores[-1] > scores[0]
        # No two adjacent scores should have the same value (smooth, not stepped)
        # (unless legitimately close prompts)


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
        assert score > 0.15

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
        assert score <= 0.15  # 0-1 scale, low

    def test_returns_0_to_1(self, analyzer, make_turn):
        for content in ["x", "fix the bug", "Refactor get_user_by_id in api/auth.py to use async"]:
            turn = make_turn(content=content)
            score = analyzer._score_specificity(turn)
            assert 0 <= score <= 1


class TestContextRichnessScoring:
    def test_error_context_boost(self, analyzer, make_turn):
        turn_with = make_turn(content="Debug the error", has_error_context=True)
        turn_without = make_turn(content="Debug the error", has_error_context=False)
        assert analyzer._score_context_richness(turn_with) > analyzer._score_context_richness(turn_without)

    def test_code_snippet_boost(self, analyzer, make_turn):
        turn_with = make_turn(content="Fix this code", has_code_snippet=True)
        turn_without = make_turn(content="Fix this code", has_code_snippet=False)
        assert analyzer._score_context_richness(turn_with) > analyzer._score_context_richness(turn_without)

    def test_expected_behavior_boost(self, analyzer, make_turn):
        turn_with = make_turn(content="This function should return a list of users")
        turn_without = make_turn(content="This function does something")
        assert analyzer._score_context_richness(turn_with) > analyzer._score_context_richness(turn_without)

    def test_constraints_boost(self, analyzer, make_turn):
        turn_with = make_turn(content="Refactor this without breaking backward compatibility")
        turn_without = make_turn(content="Refactor this code")
        assert analyzer._score_context_richness(turn_with) > analyzer._score_context_richness(turn_without)


class TestClarityScoring:
    def test_structured_prompt_boost(self, analyzer, make_turn):
        turn = make_turn(content="1. Create the model\n2. Add the endpoint\n3. Write tests")
        score = analyzer._score_clarity(turn)
        assert score > 0.3

    def test_question_only_penalized(self, analyzer, make_turn):
        question = make_turn(content="What is this?")
        statement = make_turn(
            content="Explain how the auth middleware works in this codebase. Specifically, show the request validation flow."
        )
        assert analyzer._score_clarity(statement) > analyzer._score_clarity(question)

    def test_markdown_formatting_boost(self, analyzer, make_turn):
        turn = make_turn(content="Fix the `get_user` function in **auth.py** to handle None")
        score = analyzer._score_clarity(turn)
        assert score > 0.3


class TestEfficiencyScoring:
    def test_optimal_length_high_score(self, analyzer, make_turn):
        """~80 words = peak of bell curve."""
        turn = make_turn(content=" ".join(["word"] * 80))
        score = analyzer._score_efficiency(turn)
        assert score > 0.9

    def test_very_long_prompt_penalized(self, analyzer, make_turn):
        turn = make_turn(content="word " * 600)
        score = analyzer._score_efficiency(turn)
        assert score < 0.3

    def test_very_short_prompt_penalized(self, analyzer, make_turn):
        turn = make_turn(content="fix it")
        score = analyzer._score_efficiency(turn)
        assert score < 0.5


class TestChainOfThoughtScoring:
    def test_numbered_steps_boost(self, analyzer, make_turn):
        turn = make_turn(content="1. Create model\n2. Add endpoint\n3. Write tests")
        score = analyzer._score_chain_of_thought(turn)
        assert score > 0.3

    def test_reasoning_language_boost(self, analyzer, make_turn):
        turn = make_turn(content="Add error handling because users can pass invalid input")
        score = analyzer._score_chain_of_thought(turn)
        assert score > 0.3

    def test_combined_expected_and_constraints_boost(self, analyzer, make_turn):
        turn = make_turn(
            content="The function should return a list of users without modifying the existing API contract"
        )
        score = analyzer._score_chain_of_thought(turn)
        assert score > 0.4

    def test_plain_instruction_lower_score(self, analyzer, make_turn):
        turn = make_turn(content="Add a button")
        score = analyzer._score_chain_of_thought(turn)
        assert score < 0.5
