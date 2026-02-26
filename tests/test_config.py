"""Tests for configuration schema and defaults."""

import pytest
from pydantic import ValidationError

from sparkey_reflect.config.defaults import (
    DEFAULT_ANALYSIS_DAYS,
    DEFAULT_MODEL,
    DEFAULT_PRESET,
    RETENTION_DAYS,
    SCORE_EXCELLENT,
    SCORE_GOOD,
    SCORE_FAIR,
    SCORE_POOR,
)
from sparkey_reflect.config.schema import ReflectConfig


class TestDefaults:
    def test_score_thresholds_ordering(self):
        assert SCORE_EXCELLENT > SCORE_GOOD > SCORE_FAIR > SCORE_POOR

    def test_default_analysis_days(self):
        assert DEFAULT_ANALYSIS_DAYS == 7

    def test_retention_days_reasonable(self):
        assert 30 <= RETENTION_DAYS <= 730

    def test_default_model_set(self):
        assert DEFAULT_MODEL is not None
        assert len(DEFAULT_MODEL) > 0

    def test_default_preset_set(self):
        assert DEFAULT_PRESET == "coaching"


class TestReflectConfig:
    def test_defaults(self):
        config = ReflectConfig()
        assert config.default_days == DEFAULT_ANALYSIS_DAYS
        assert config.retention_days == RETENTION_DAYS
        assert config.sync_enabled is False
        assert config.model == DEFAULT_MODEL

    def test_custom_values(self):
        config = ReflectConfig(
            default_tool="claude-code",
            default_days=14,
            retention_days=365,
        )
        assert config.default_tool == "claude-code"
        assert config.default_days == 14
        assert config.retention_days == 365

    def test_days_validation_min(self):
        with pytest.raises(ValidationError):
            ReflectConfig(default_days=0)

    def test_days_validation_max(self):
        with pytest.raises(ValidationError):
            ReflectConfig(default_days=400)

    def test_retention_validation_min(self):
        with pytest.raises(ValidationError):
            ReflectConfig(retention_days=10)

    def test_retention_validation_max(self):
        with pytest.raises(ValidationError):
            ReflectConfig(retention_days=800)

    def test_get_db_path_default(self):
        config = ReflectConfig()
        path = config.get_db_path()
        assert path.name == "reflect.db"
        assert "sparkey" in str(path)

    def test_get_db_path_custom(self):
        config = ReflectConfig(db_path="/tmp/custom.db")
        assert str(config.get_db_path()) == "/tmp/custom.db"
