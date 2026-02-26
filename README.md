# Sparkey Reflect

**AI coding advisor** that analyzes your AI coding sessions (Claude Code, Cursor, GitHub Copilot) and generates personalized coaching insights. Runs 100% locally — your conversation data never leaves your machine.

## Install

### As a Claude Code plugin

```bash
# From the Claude Code plugin marketplace
/plugin marketplace add sparkey-ai/sparkey-reflect

# Or install directly
/plugin install sparkey@sparkey-reflect
```

### As a CLI tool

```bash
pip install sparkey-reflect
```

## Usage

### Via Claude Code (recommended)

```
/sparkey:reflect              # Weekly report (last 7 days)
/sparkey:reflect daily        # Yesterday's sessions
/sparkey:reflect monthly      # Last 30 days
/sparkey:reflect full         # Full quarter (90 days)
/sparkey:reflect 14           # Custom day count
/sparkey:reflect deep-dive prompt_engineering   # Skill deep-dive
/sparkey:reflect update-rules # Improve your CLAUDE.md
```

### Via CLI

```bash
sparkey-reflect analyze --tool claude-code --days 7
sparkey-reflect analyze --tool cursor --days 30 --format json
sparkey-reflect status
sparkey-reflect rules
sparkey-reflect trends --days 30
sparkey-reflect learning-path
```

## Parameters

| Parameter | Description |
|-----------|-------------|
| `daily` | Analyze yesterday's sessions |
| `weekly` *(default)* | Last 7 days |
| `monthly` | Last 30 days |
| `full` | Full quarter — 90 days |
| `<number>` | Custom number of days |
| `deep-dive <skill>` | In-depth analysis of one skill |
| `update-rules` | Improve CLAUDE.md based on insights |

### Deep-dive skills

| Skill | What it analyzes |
|-------|-----------------|
| `prompt_engineering` | Prompt clarity, specificity, structure |
| `context_management` | How you manage context window and file references |
| `tool_mastery` | Usage of available tools and features |
| `conversation_flow` | Session structure and conversation patterns |
| `session_habits` | Session length, time-of-day, and workflow patterns |
| `rule_file_quality` | CLAUDE.md and instruction file effectiveness |
| `outcome_quality` | Success rate and task completion quality |

## Supported AI tools

- **Claude Code** — full support (conversations, tool usage, rule files)
- **Cursor** — session analysis
- **GitHub Copilot** — usage metrics

## Teams edition

Get version code + project management integrations, trend tracking, usage analytics, and industry benchmarks at [sparkey.ai](https://sparkey.ai).

## License

MIT
