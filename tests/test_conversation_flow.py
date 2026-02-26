"""Tests for the Conversation Flow analyzer."""

import pytest

from sparkey_reflect.analyzers.conversation_flow import ConversationFlowAnalyzer


@pytest.fixture
def analyzer():
    return ConversationFlowAnalyzer()


class TestConversationFlowAnalyzer:
    def test_key_and_name(self, analyzer):
        assert analyzer.get_key() == "conversation_flow"
        assert analyzer.get_name() == "Conversation Flow"

    def test_empty_sessions(self, analyzer):
        result = analyzer.analyze([])
        assert result.score == 0

    def test_perfect_session(self, analyzer, make_session, make_turn):
        """Single-turn session = good flow."""
        session = make_session(turns=[
            make_turn(content="Add a logout button to the navbar"),
        ])
        result = analyzer.analyze([session])
        # Single turn = good turns score, 1.0 first acceptance
        assert result.score > 40

    def test_high_correction_rate_lowers_score(self, analyzer, make_session, make_turn):
        session = make_session(turns=[
            make_turn(content="Add a button"),
            make_turn(role="assistant", content="Done"),
            make_turn(content="No, wrong, that's not what I meant"),
            make_turn(role="assistant", content="Let me try again"),
            make_turn(content="Try again, undo that"),
            make_turn(role="assistant", content="Done"),
        ])
        result = analyzer.analyze([session])
        assert result.metrics["correction_rate"] > 0.3

    def test_context_restatement_detected(self, analyzer, make_session, make_turn):
        session = make_session(turns=[
            make_turn(content="Fix the auth bug"),
            make_turn(role="assistant", content="Which one?"),
            make_turn(content="As I said, the error I showed you from before"),
            make_turn(role="assistant", content="Got it"),
        ])
        result = analyzer.analyze([session])
        assert result.metrics["context_loss_rate"] > 0

    def test_first_acceptance_detected(self, analyzer, make_session, make_turn):
        session = make_session(turns=[
            make_turn(content="Add a test"),
            make_turn(role="assistant", content="Here's the test"),
            make_turn(content="Perfect, looks good, thanks"),
        ])
        result = analyzer.analyze([session])
        assert result.metrics["first_response_acceptance"] > 0.5

    def test_multiple_sessions_averaged(self, analyzer, sample_sessions):
        result = analyzer.analyze(sample_sessions)
        assert result.score > 0
        assert result.session_count == 3
        assert "avg_turns_to_resolution" in result.metrics

    def test_score_bounded(self, analyzer, sample_sessions):
        result = analyzer.analyze(sample_sessions)
        assert 0 <= result.score <= 100

    def test_iteration_velocity_metric(self, analyzer, make_session, make_turn):
        """New iteration velocity dimension is computed."""
        session = make_session(turns=[
            make_turn(content="Add auth"),
            make_turn(role="assistant", content="I'll implement the full authentication system with login, logout, and session management. Here's the complete implementation..."),
        ])
        result = analyzer.analyze([session])
        assert "iteration_velocity" in result.metrics
        # Assistant produced much more than user -> high velocity
        assert result.metrics["iteration_velocity"] > 1.0
