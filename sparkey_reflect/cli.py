"""
Sparkey Reflect CLI

Command-line interface for running local AI coding advisor analysis.

Usage:
    sparkey-reflect analyze [--tool TOOL] [--days N] [--preset PRESET]
    sparkey-reflect report [--period weekly|monthly] [--format text|json]
    sparkey-reflect trends [--metric KEY] [--days N]
    sparkey-reflect rules [--workspace PATH]
    sparkey-reflect status
"""

import json
import logging
import sys
from datetime import datetime, timezone

import click

from sparkey_reflect.core.models import ToolType


def _setup_logging(verbose: bool):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )


def _resolve_tool(tool_str: str | None) -> ToolType | None:
    if tool_str is None:
        return None
    mapping = {
        "claude-code": ToolType.CLAUDE_CODE,
        "claude_code": ToolType.CLAUDE_CODE,
        "claude": ToolType.CLAUDE_CODE,
        "cursor": ToolType.CURSOR,
        "copilot": ToolType.COPILOT,
    }
    result = mapping.get(tool_str.lower())
    if result is None:
        raise click.BadParameter(
            f"Unknown tool: {tool_str}. Supported: claude-code, cursor, copilot"
        )
    return result


@click.group()
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging")
def cli(verbose: bool):
    """Sparkey Reflect - AI Coding Advisor"""
    _setup_logging(verbose)


@cli.command()
@click.option("--tool", "-t", default=None, help="AI tool (claude-code, cursor, copilot)")
@click.option("--days", "-d", default=7, help="Days of history to analyze")
@click.option("--preset", "-p", default="coaching", help="Analyzer preset (quick, coaching, full)")
@click.option("--workspace", "-w", default=None, help="Workspace path filter")
@click.option("--format", "output_format", default="text", type=click.Choice(["text", "json", "prompt"]))
@click.option("--no-llm", is_flag=True, help="Skip LLM call, return scores only")
def analyze(tool, days, preset, workspace, output_format, no_llm):
    """Analyze AI coding sessions."""
    from sparkey_reflect.core.engine import ReflectEngine

    # prompt format: output raw LLM prompt for use as a slash command
    if output_format == "prompt":
        no_llm = True

    engine = ReflectEngine(use_llm=not no_llm)

    try:
        tool_type = _resolve_tool(tool)
        report = engine.analyze(
            tool=tool_type,
            days=days,
            preset=preset,
            workspace_path=workspace,
        )
    except RuntimeError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    if output_format == "json":
        click.echo(json.dumps(report.to_dict(), indent=2))
    elif output_format == "prompt":
        _output_llm_prompt(engine, report, tool_type or engine._auto_detect_tool(), days)
    else:
        generator = engine.insight_generator
        if days <= 1:
            click.echo(generator.format_daily_digest(report))
        else:
            click.echo(generator.format_weekly_digest(report))


def _output_llm_prompt(engine, report, tool_type, days):
    """Output the full LLM prompt (system + user) for use as a Claude Code slash command."""
    from datetime import timedelta, timezone as tz
    from sparkey_reflect.insights.llm_generator import LLMInsightGenerator

    llm_gen = LLMInsightGenerator()
    reader = engine._get_reader(tool_type)
    sessions = reader.read_sessions(
        since=report.period_start,
        until=report.period_end,
    )
    rule_files = reader.read_rule_files()
    trends = report.trends or {}

    system_prompt = llm_gen._load_system_prompt()
    user_prompt = llm_gen._build_user_prompt(
        report.results, sessions, rule_files, trends,
    )

    click.echo(system_prompt)
    click.echo("\n---\n")
    click.echo(user_prompt)


@cli.command()
@click.option("--tool", "-t", default=None, help="AI tool")
@click.option("--period", default="weekly", type=click.Choice(["daily", "weekly", "monthly"]))
@click.option("--format", "output_format", default="text", type=click.Choice(["text", "json"]))
def report(tool, period, output_format):
    """Show the latest analysis report."""
    from sparkey_reflect.core.engine import ReflectEngine

    period_days = {"daily": 1, "weekly": 7, "monthly": 30}[period]
    engine = ReflectEngine()

    try:
        tool_type = _resolve_tool(tool)
        report_data = engine.analyze(
            tool=tool_type,
            days=period_days,
            preset="coaching",
        )
    except RuntimeError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    if output_format == "json":
        click.echo(json.dumps(report_data.to_dict(), indent=2))
    else:
        generator = engine.insight_generator
        if period == "daily":
            click.echo(generator.format_daily_digest(report_data))
        else:
            click.echo(generator.format_weekly_digest(report_data))


@cli.command(name="learning-path")
@click.option("--tool", "-t", default=None, help="AI tool")
@click.option("--days", "-d", default=7, help="Days of history to analyze")
@click.option("--format", "output_format", default="text", type=click.Choice(["text", "json"]))
@click.option("--no-llm", is_flag=True, help="Skip LLM call, return scores only")
def learning_path(tool, days, output_format, no_llm):
    """Generate a personalized learning path."""
    from sparkey_reflect.core.engine import ReflectEngine
    from sparkey_reflect.insights.learning_paths import LearningPathBuilder
    from sparkey_reflect.insights.llm_generator import LLMInsightGenerator

    engine = ReflectEngine(use_llm=not no_llm)

    try:
        tool_type = _resolve_tool(tool)
        report = engine.analyze(tool=tool_type, days=days, preset="full")
    except RuntimeError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    builder = LearningPathBuilder()

    # If LLM insights are available in the report, use them for richer paths
    if not no_llm and report.insights:
        # Re-generate LLM data to get learning_path section
        llm_gen = LLMInsightGenerator()
        # Read sessions + rule files via the engine's reader
        try:
            reader = engine._get_reader(tool_type or engine._auto_detect_tool())
            from datetime import timedelta, timezone as tz
            since = report.period_start
            sessions = reader.read_sessions(since=since, until=report.period_end)
            rule_files = reader.read_rule_files()
            llm_data = llm_gen.generate_insights(
                results=report.results,
                sessions=sessions,
                rule_files=rule_files,
            )
            skill_areas = builder.build_from_llm(llm_data, report.results)
        except Exception:
            skill_areas = builder.build(report.results)
    else:
        skill_areas = builder.build(report.results)

    if output_format == "json":
        data = [
            {
                "name": sa.name,
                "key": sa.key,
                "score": round(sa.score, 1),
                "target": sa.target,
                "deficit": round(sa.deficit, 1),
                "current_level": sa.current_level,
                "recommendations": sa.recommendations,
            }
            for sa in skill_areas
        ]
        click.echo(json.dumps(data, indent=2))
    else:
        click.echo(builder.format(skill_areas))


@cli.command()
@click.option("--tool", "-t", default=None, help="AI tool")
@click.option("--metric", "-m", default=None, help="Specific metric key")
@click.option("--days", "-d", default=30, help="Days of trend history")
@click.option("--format", "output_format", default="text", type=click.Choice(["text", "json"]))
def trends(tool, metric, days, output_format):
    """Show improvement trends over time."""
    from sparkey_reflect.core.engine import ReflectEngine

    engine = ReflectEngine()

    try:
        tool_type = _resolve_tool(tool)
        trend_data = engine.get_trends(tool=tool_type, metric_key=metric, days=days)
    except RuntimeError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    if output_format == "json":
        click.echo(json.dumps(trend_data, indent=2, default=str))
    else:
        if not trend_data:
            click.echo("No trend data available yet. Run 'sparkey-reflect analyze' first.")
            return

        for key, points in trend_data.items():
            click.echo(f"\n  {key}:")
            if not points:
                click.echo("    No data")
                continue
            for pt in points[-10:]:  # last 10 points
                date = pt.get("measured_at", "?")[:10]
                val = pt.get("metric_value", 0)
                bar = "#" * int(val / 5)
                click.echo(f"    {date}  {bar} {val:.1f}")


@cli.command()
@click.option("--tool", "-t", default=None, help="AI tool")
@click.option("--workspace", "-w", default=None, help="Workspace path")
@click.option("--format", "output_format", default="text", type=click.Choice(["text", "json"]))
def rules(tool, workspace, output_format):
    """Analyze rule and instruction files."""
    from sparkey_reflect.core.engine import ReflectEngine

    engine = ReflectEngine()

    try:
        tool_type = _resolve_tool(tool)
        rule_files = engine.analyze_rules(tool=tool_type, workspace_path=workspace)
    except RuntimeError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    if output_format == "json":
        data = [
            {
                "file_path": r.file_path,
                "file_type": r.file_type,
                "exists": r.exists,
                "word_count": r.word_count,
                "section_count": r.section_count,
                "sections": r.sections,
                "has_examples": r.has_examples,
                "has_constraints": r.has_constraints,
                "has_project_context": r.has_project_context,
                "has_style_guide": r.has_style_guide,
            }
            for r in rule_files
        ]
        click.echo(json.dumps(data, indent=2))
    else:
        found = [r for r in rule_files if r.exists]
        missing = [r for r in rule_files if not r.exists]

        if found:
            click.echo("\n  Found rule files:")
            for r in found:
                check = lambda b: "Y" if b else "N"
                click.echo(
                    f"    {r.file_type:<25} {r.word_count:>5} words  "
                    f"sections={r.section_count}  "
                    f"examples={check(r.has_examples)}  "
                    f"constraints={check(r.has_constraints)}  "
                    f"context={check(r.has_project_context)}"
                )
        if missing:
            click.echo("\n  Missing (recommended):")
            for r in missing:
                click.echo(f"    {r.file_type:<25} {r.file_path}")


@cli.command()
def status():
    """Show available AI tools and data ranges."""
    from sparkey_reflect.core.engine import ReflectEngine

    engine = ReflectEngine()
    statuses = engine.get_status()

    click.echo("\n  Sparkey Reflect - Tool Status\n")
    for s in statuses:
        icon = "+" if s["available"] else "-"
        click.echo(f"  {icon} {s['tool']}")
        if s["available"]:
            click.echo(f"      Data range: {s['earliest_data']} to {s['latest_data']}")
            for loc in s.get("data_locations", []):
                click.echo(f"      Location: {loc}")
        else:
            click.echo("      Not available")

    click.echo()


@cli.command()
@click.option("--key", "-k", help="Config key to get/set")
@click.option("--value", "-V", help="Config value to set")
def config(key, value):
    """View or update Reflect configuration."""
    from sparkey_reflect.core.storage import ReflectStorage

    storage = ReflectStorage()

    if key and value:
        storage.set_config(key, value)
        click.echo(f"Set {key} = {value}")
    elif key:
        val = storage.get_config(key)
        if val:
            click.echo(f"{key} = {val}")
        else:
            click.echo(f"{key} is not set")
    else:
        click.echo("\n  Sparkey Reflect Configuration")
        click.echo(f"  DB path: {storage.db_path}")
        click.echo(f"  Retention: {storage.get_config('retention_days', '180')} days")
        click.echo(f"  Default tool: {storage.get_config('default_tool', 'auto-detect')}")
        click.echo(f"  Sync enabled: {storage.get_config('sync_enabled', 'false')}")
        click.echo()


if __name__ == "__main__":
    cli()
