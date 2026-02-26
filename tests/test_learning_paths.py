"""Tests for the Learning Path Builder."""

import pytest

from sparkey_reflect.core.models import AnalysisResult
from sparkey_reflect.insights.learning_paths import LearningPathBuilder, SkillArea


@pytest.fixture
def builder():
    return LearningPathBuilder()


class TestLearningPathBuilder:
    def test_build_from_results(self, builder, make_result):
        results = [
            make_result(key="prompt_quality", name="Prompt Quality", score=80.0),
            make_result(key="conversation_flow", name="Conversation Flow", score=40.0),
            make_result(key="session_patterns", name="Session Patterns", score=60.0),
        ]
        skill_areas = builder.build(results)
        assert len(skill_areas) == 3
        # Sorted by deficit (highest first)
        assert skill_areas[0].key == "conversation_flow"
        assert skill_areas[0].deficit == 35.0
        assert skill_areas[-1].key == "prompt_quality"
        assert skill_areas[-1].deficit == 0  # above target

    def test_build_score_above_target(self, builder, make_result):
        results = [make_result(score=90.0)]
        skill_areas = builder.build(results)
        assert skill_areas[0].deficit == 0  # no deficit when above target

    def test_build_from_llm(self, builder, make_result):
        llm_data = {
            "learning_path": [
                {
                    "skill_name": "Prompt Quality",
                    "current_level": "intermediate",
                    "recommendations": ["Be more specific", "Include file refs"],
                },
                {
                    "skill_name": "Context Management",
                    "current_level": "beginner",
                    "recommendations": ["Always include error context"],
                },
            ]
        }
        results = [
            make_result(key="prompt_quality", score=65.0),
            make_result(key="context_management", name="Context Management", score=35.0),
        ]
        skill_areas = builder.build_from_llm(llm_data, results)
        assert len(skill_areas) == 2
        # context_management has higher deficit, should be first
        assert skill_areas[0].key == "context_management"
        assert skill_areas[0].current_level == "beginner"
        assert len(skill_areas[0].recommendations) == 1

    def test_format_empty(self, builder):
        result = builder.format([])
        assert "No learning path data" in result

    def test_format_on_track(self, builder):
        skill_areas = [
            SkillArea(name="Prompt Quality", key="pq", score=80, deficit=0),
        ]
        result = builder.format(skill_areas)
        assert "On track" in result

    def test_format_has_gap(self, builder):
        skill_areas = [
            SkillArea(
                name="Flow",
                key="flow",
                score=30,
                deficit=45,
                recommendations=["Practice shorter sessions"],
            ),
        ]
        result = builder.format(skill_areas)
        assert "Gap:" in result
        assert "Practice shorter sessions" in result

    def test_format_close_gap(self, builder):
        skill_areas = [
            SkillArea(name="Tools", key="tools", score=65, deficit=10),
        ]
        result = builder.format(skill_areas)
        assert "Close" in result
