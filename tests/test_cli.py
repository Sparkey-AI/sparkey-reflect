"""Tests for the CLI using Click's test runner."""

from click.testing import CliRunner

from sparkey_reflect.cli import cli, _resolve_tool
from sparkey_reflect.core.models import ToolType


class TestResolveTool:
    def test_claude_code(self):
        assert _resolve_tool("claude-code") == ToolType.CLAUDE_CODE

    def test_claude_shorthand(self):
        assert _resolve_tool("claude") == ToolType.CLAUDE_CODE

    def test_cursor(self):
        assert _resolve_tool("cursor") == ToolType.CURSOR

    def test_copilot(self):
        assert _resolve_tool("copilot") == ToolType.COPILOT

    def test_case_insensitive(self):
        assert _resolve_tool("Claude-Code") == ToolType.CLAUDE_CODE
        assert _resolve_tool("CURSOR") == ToolType.CURSOR

    def test_none_returns_none(self):
        assert _resolve_tool(None) is None

    def test_unknown_raises(self):
        import click
        import pytest
        with pytest.raises(click.BadParameter, match="Unknown tool"):
            _resolve_tool("vscode")


class TestCLI:
    def test_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Sparkey Reflect" in result.output

    def test_status_command(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["status"])
        assert result.exit_code == 0
        assert "Tool Status" in result.output

    def test_config_command_no_args(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["config"])
        assert result.exit_code == 0
        assert "Configuration" in result.output

    def test_analyze_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["analyze", "--help"])
        assert result.exit_code == 0
        assert "--tool" in result.output
        assert "--days" in result.output

    def test_rules_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["rules", "--help"])
        assert result.exit_code == 0

    def test_trends_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["trends", "--help"])
        assert result.exit_code == 0

    def test_learning_path_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["learning-path", "--help"])
        assert result.exit_code == 0
