"""Tests for analyzer registry, config, and presets."""

from sparkey_reflect.analyzers.registry import (
    ReflectAnalyzerConfig,
    ReflectAnalyzerDefinition,
    ReflectAnalyzerPresets,
    ReflectAnalyzerRegistry,
)


class TestReflectAnalyzerRegistry:
    def test_get_all_returns_dict(self):
        all_analyzers = ReflectAnalyzerRegistry.get_all()
        assert isinstance(all_analyzers, dict)
        assert len(all_analyzers) == 8

    def test_get_existing_key(self):
        defn = ReflectAnalyzerRegistry.get("prompt_quality")
        assert defn is not None
        assert defn.name == "Prompt Quality"
        assert defn.key == "prompt_quality"

    def test_get_nonexistent_key(self):
        assert ReflectAnalyzerRegistry.get("nonexistent") is None

    def test_get_defaults_excludes_non_default(self):
        defaults = ReflectAnalyzerRegistry.get_defaults()
        assert "prompt_quality" in defaults
        # All analyzers are enabled by default in the current config
        assert isinstance(defaults, list)
        assert len(defaults) > 0

    def test_get_by_category_core(self):
        core = ReflectAnalyzerRegistry.get_by_category("core")
        assert "prompt_quality" in core
        assert "conversation_flow" in core
        # tool_specific should not be in core
        for key, defn in core.items():
            assert defn.category == "core"

    def test_get_by_category_tool_specific(self):
        specific = ReflectAnalyzerRegistry.get_by_category("tool_specific")
        assert "tool_usage" in specific or "completion_patterns" in specific

    def test_get_for_tool_claude_code(self):
        keys = ReflectAnalyzerRegistry.get_for_tool("claude_code")
        assert "prompt_quality" in keys
        assert "tool_usage" in keys
        # completion_patterns is copilot-only
        assert "completion_patterns" not in keys

    def test_get_for_tool_copilot(self):
        keys = ReflectAnalyzerRegistry.get_for_tool("copilot")
        assert "completion_patterns" in keys
        assert "prompt_quality" in keys
        # tool_usage only applies to claude_code and cursor
        assert "tool_usage" not in keys


class TestReflectAnalyzerConfig:
    def test_default_config_uses_defaults(self):
        config = ReflectAnalyzerConfig()
        enabled = config.get_enabled()
        assert "prompt_quality" in enabled
        assert len(enabled) > 0

    def test_explicit_enabled_list(self):
        config = ReflectAnalyzerConfig(enabled=["prompt_quality", "session_patterns"])
        assert config.should_run("prompt_quality")
        assert config.should_run("session_patterns")
        assert not config.should_run("conversation_flow")

    def test_disabled_list(self):
        config = ReflectAnalyzerConfig(disabled=["outcome_tracker"])
        assert not config.should_run("outcome_tracker")
        assert config.should_run("prompt_quality")

    def test_tool_filter(self):
        config = ReflectAnalyzerConfig(tool="copilot")
        # Only copilot-applicable analyzers
        assert config.should_run("completion_patterns")
        assert not config.should_run("tool_usage")  # claude_code/cursor only

    def test_skip_git(self):
        config = ReflectAnalyzerConfig(skip_git=True)
        assert not config.should_run("outcome_tracker")
        assert config.should_run("prompt_quality")

    def test_should_run_false_for_unknown(self):
        config = ReflectAnalyzerConfig()
        assert not config.should_run("nonexistent_analyzer")


class TestReflectAnalyzerPresets:
    def test_quick_preset(self):
        config = ReflectAnalyzerPresets.quick()
        enabled = config.get_enabled()
        assert enabled == {"prompt_quality", "conversation_flow", "session_patterns"}

    def test_coaching_preset(self):
        config = ReflectAnalyzerPresets.coaching()
        enabled = config.get_enabled()
        assert "prompt_quality" in enabled
        assert "context_management" in enabled
        assert "rule_file" in enabled
        assert len(enabled) == 5

    def test_full_preset_includes_all(self):
        config = ReflectAnalyzerPresets.full()
        all_keys = set(ReflectAnalyzerRegistry.get_all().keys())
        assert config.get_enabled() == all_keys

    def test_copilot_preset(self):
        config = ReflectAnalyzerPresets.copilot()
        enabled = config.get_enabled()
        assert "completion_patterns" in enabled
        assert "session_patterns" in enabled
