"""
Default Configuration

Centralized defaults for the Reflect engine.
"""

from pathlib import Path

# Storage
DEFAULT_DB_DIR = Path.home() / ".sparkey" / "reflect"
DEFAULT_DB_PATH = DEFAULT_DB_DIR / "reflect.db"
RETENTION_DAYS = 180

# Analysis windows
DEFAULT_ANALYSIS_DAYS = 7
DAILY_WINDOW_DAYS = 1
WEEKLY_WINDOW_DAYS = 7
MONTHLY_WINDOW_DAYS = 30
QUARTERLY_WINDOW_DAYS = 90

# Analyzer presets
DEFAULT_PRESET = "coaching"

# Claude Code paths
CLAUDE_DIR = Path.home() / ".claude"
CLAUDE_PROJECTS_DIR = CLAUDE_DIR / "projects"

# Cursor paths
CURSOR_STORAGE_DIR = Path.home() / "Library" / "Application Support" / "Cursor" / "User" / "workspaceStorage"

# Copilot paths
VSCODE_LOGS_DIR = Path.home() / "Library" / "Application Support" / "Code" / "logs"
COPILOT_TRACES_DIR = DEFAULT_DB_DIR / "copilot_traces"

# Score thresholds
SCORE_EXCELLENT = 80
SCORE_GOOD = 60
SCORE_FAIR = 40
SCORE_POOR = 20

# LLM Configuration
DEFAULT_MODEL = "claude-sonnet-4-6"
MAX_OUTPUT_TOKENS = 16384
CONTEXT_WINDOW_LIMIT = 180_000  # soft limit — only triggers summarization above this
MAX_CODE_BLOCK_LINES = 5  # code blocks > this get truncated in extraction
