"""
Shared Scoring Primitives

Reusable curve functions for smooth, industry-benchmarked scoring across all
analyzers. Replaces step-function (if/elif cascade) scoring with continuous
mathematical curves grounded in DORA, SPACE, DevEx, GitClear, and METR research.
"""

import math
from typing import List, Tuple


def sigmoid(x: float, midpoint: float, steepness: float = 1.0) -> float:
    """S-curve: 0->1. Ideal for rate-based metrics (0.0-1.0 inputs).

    Args:
        x: Input value.
        midpoint: Value where output is 0.5.
        steepness: Higher = sharper transition around midpoint.
    """
    z = -steepness * (x - midpoint)
    # Clamp to avoid overflow
    z = max(-500, min(500, z))
    return 1.0 / (1.0 + math.exp(z))


def bell(x: float, center: float, width: float) -> float:
    """Gaussian bell: peaks at center, decays by width.

    Ideal for optimal-range metrics where both too-low and too-high are bad.

    Args:
        x: Input value.
        center: Peak value (output = 1.0).
        width: Standard deviation controlling decay speed.
    """
    if width == 0:
        return 1.0 if x == center else 0.0
    return math.exp(-0.5 * ((x - center) / width) ** 2)


def linear_clamp(x: float, low: float, high: float) -> float:
    """Linear interpolation 0->1 clamped at [low, high].

    Args:
        x: Input value.
        low: Value at which output is 0.
        high: Value at which output is 1.
    """
    if high <= low:
        return 1.0 if x >= high else 0.0
    if x <= low:
        return 0.0
    if x >= high:
        return 1.0
    return (x - low) / (high - low)


def diminishing(x: float, scale: float = 1.0) -> float:
    """Square-root diminishing returns: fast start, slow ceiling.

    Args:
        x: Input value (non-negative).
        scale: Value at which output reaches 1.0.
    """
    if scale <= 0:
        return 1.0
    return min(1.0, math.sqrt(max(0, x) / scale))


def count_score(n: int, thresholds: List[Tuple[int, float]]) -> float:
    """Piecewise score for discrete counts.

    Args:
        n: Input count.
        thresholds: List of (count, score) tuples. Returns highest score
                    whose threshold is met.
    """
    result = 0.0
    for threshold, value in sorted(thresholds):
        if n >= threshold:
            result = value
    return result


def weighted_sum(dimensions: List[Tuple[float, float]]) -> float:
    """Compute weighted sum of (score_0_to_1, weight) pairs. Returns 0-100.

    Args:
        dimensions: List of (score, weight) tuples where score is 0.0-1.0
                    and weight is the relative importance.
    """
    total_w = sum(w for _, w in dimensions)
    if total_w == 0:
        return 0.0
    return 100.0 * sum(s * w for s, w in dimensions) / total_w
