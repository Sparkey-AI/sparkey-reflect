"""
Learning Path Builder

Builds personalized learning paths from LLM-generated data or analyzer scores.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from sparkey_reflect.core.models import AnalysisResult
from sparkey_reflect.insights.templates import LEARNING_PATH_TEMPLATE


@dataclass
class SkillArea:
    """A learnable skill area with score and recommendations."""
    name: str
    key: str
    score: float = 0.0
    target: float = 75.0
    deficit: float = 0.0
    current_level: str = "intermediate"
    recommendations: List[str] = field(default_factory=list)


class LearningPathBuilder:
    """Builds personalized learning paths from LLM-generated data."""

    def build_from_llm(self, llm_data: Dict[str, Any], results: List[AnalysisResult]) -> List[SkillArea]:
        """
        Build a learning path from LLM-generated learning_path data,
        enriched with analyzer scores.
        """
        score_map = {r.analyzer_key: r.score for r in results}
        skill_areas = []

        for item in llm_data.get("learning_path", []):
            name = item.get("skill_name", "Unknown")
            recs = item.get("recommendations", [])
            level = item.get("current_level", "intermediate")

            # Try to find a matching score
            score = 0.0
            key = name.lower().replace(" ", "_")
            for ak, av in score_map.items():
                if ak in key or key in ak:
                    score = av
                    break

            target = 75.0
            deficit = max(0, target - score)

            skill_areas.append(SkillArea(
                name=name,
                key=key,
                score=score,
                target=target,
                deficit=deficit,
                current_level=level,
                recommendations=recs,
            ))

        # Sort by deficit descending
        skill_areas.sort(key=lambda s: s.deficit, reverse=True)
        return skill_areas

    def build(self, results: List[AnalysisResult]) -> List[SkillArea]:
        """Build a basic learning path from scores only (no LLM)."""
        skill_areas = []
        for result in results:
            target = 75.0
            deficit = max(0, target - result.score)
            skill_areas.append(SkillArea(
                name=result.analyzer_name,
                key=result.analyzer_key,
                score=result.score,
                target=target,
                deficit=deficit,
            ))
        skill_areas.sort(key=lambda s: s.deficit, reverse=True)
        return skill_areas

    def format(self, skill_areas: List[SkillArea]) -> str:
        """Format learning path as readable text."""
        if not skill_areas:
            return "No learning path data available. Run an analysis first."

        priority_lines = []
        for i, area in enumerate(skill_areas, 1):
            if area.deficit <= 0:
                icon = "+"
                status = "On track"
            elif area.deficit < 20:
                icon = "~"
                status = "Close"
            else:
                icon = "!"
                status = f"Gap: {area.deficit:.0f} pts"
            level_str = f" [{area.current_level}]" if area.current_level else ""
            priority_lines.append(
                f"  {i}. {icon} {area.name:<25} "
                f"Score: {area.score:.0f}/100  Target: {area.target:.0f}  {status}{level_str}"
            )

        step_lines = []
        for area in skill_areas[:5]:
            if area.deficit <= 0 and not area.recommendations:
                continue
            if area.recommendations:
                step_lines.append(f"\n  **{area.name}**:")
                for rec in area.recommendations:
                    step_lines.append(f"    - {rec}")

        if not step_lines:
            step_lines = ["  All skill areas are on track! Keep up the great work."]

        return LEARNING_PATH_TEMPLATE.format(
            priority_areas="\n".join(priority_lines),
            next_steps="\n".join(step_lines),
        )
