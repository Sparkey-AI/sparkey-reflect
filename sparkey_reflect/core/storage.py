"""
Sparkey Reflect Local Storage

SQLite-based storage at ~/.sparkey/reflect/reflect.db
Stores computed metrics, scores, trends -- never raw conversation text.
"""

import json
import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

DEFAULT_DB_DIR = Path.home() / ".sparkey" / "reflect"
DEFAULT_DB_PATH = DEFAULT_DB_DIR / "reflect.db"
RETENTION_DAYS = 180


class ReflectStorage:
    """Local SQLite storage for Reflect analysis data."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or DEFAULT_DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: Optional[sqlite3.Connection] = None
        self._ensure_schema()

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
        return self._conn

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    # =========================================================================
    # Schema
    # =========================================================================

    def _ensure_schema(self):
        """Create tables if they don't exist."""
        with self.conn:
            self.conn.executescript("""
                CREATE TABLE IF NOT EXISTS session_metadata (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT UNIQUE NOT NULL,
                    tool TEXT NOT NULL,
                    start_time TEXT,
                    end_time TEXT,
                    duration_minutes REAL DEFAULT 0,
                    turn_count INTEGER DEFAULT 0,
                    user_turn_count INTEGER DEFAULT 0,
                    tool_use_count INTEGER DEFAULT 0,
                    total_input_tokens INTEGER DEFAULT 0,
                    total_output_tokens INTEGER DEFAULT 0,
                    session_type TEXT DEFAULT 'unknown',
                    workspace_path TEXT,
                    branch TEXT,
                    model TEXT,
                    created_at TEXT DEFAULT (datetime('now'))
                );

                CREATE TABLE IF NOT EXISTS analysis_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    analyzer_key TEXT NOT NULL,
                    score REAL NOT NULL,
                    metrics TEXT NOT NULL,  -- JSON
                    session_count INTEGER DEFAULT 0,
                    period_start TEXT NOT NULL,
                    period_end TEXT NOT NULL,
                    tool TEXT NOT NULL,
                    created_at TEXT DEFAULT (datetime('now'))
                );

                CREATE TABLE IF NOT EXISTS insights (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    category TEXT NOT NULL,
                    title TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    recommendation TEXT NOT NULL,
                    evidence TEXT,
                    metric_key TEXT,
                    metric_value REAL,
                    trend TEXT DEFAULT 'insufficient_data',
                    tool TEXT NOT NULL,
                    period_start TEXT,
                    period_end TEXT,
                    created_at TEXT DEFAULT (datetime('now'))
                );

                CREATE TABLE IF NOT EXISTS reports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    tool TEXT NOT NULL,
                    period_start TEXT NOT NULL,
                    period_end TEXT NOT NULL,
                    overall_score REAL NOT NULL,
                    session_count INTEGER DEFAULT 0,
                    total_turns INTEGER DEFAULT 0,
                    total_tokens INTEGER DEFAULT 0,
                    total_duration_minutes REAL DEFAULT 0,
                    report_data TEXT NOT NULL,  -- JSON
                    created_at TEXT DEFAULT (datetime('now'))
                );

                CREATE TABLE IF NOT EXISTS trends (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_key TEXT NOT NULL,
                    metric_value REAL NOT NULL,
                    tool TEXT NOT NULL,
                    measured_at TEXT NOT NULL,
                    period_type TEXT DEFAULT 'daily',  -- daily, weekly, monthly
                    created_at TEXT DEFAULT (datetime('now'))
                );

                CREATE TABLE IF NOT EXISTS config (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT DEFAULT (datetime('now'))
                );

                CREATE INDEX IF NOT EXISTS idx_session_tool ON session_metadata(tool);
                CREATE INDEX IF NOT EXISTS idx_session_start ON session_metadata(start_time);
                CREATE INDEX IF NOT EXISTS idx_analysis_period ON analysis_results(period_start, period_end);
                CREATE INDEX IF NOT EXISTS idx_analysis_analyzer ON analysis_results(analyzer_key);
                CREATE INDEX IF NOT EXISTS idx_insights_category ON insights(category);
                CREATE INDEX IF NOT EXISTS idx_insights_severity ON insights(severity);
                CREATE INDEX IF NOT EXISTS idx_trends_metric ON trends(metric_key, measured_at);
                CREATE INDEX IF NOT EXISTS idx_reports_period ON reports(period_start, period_end);
            """)

    # =========================================================================
    # Session Metadata
    # =========================================================================

    def save_session_metadata(self, session) -> int:
        """Save session metadata (never raw content)."""
        from sparkey_reflect.core.models import Session

        with self.conn:
            cursor = self.conn.execute(
                """INSERT OR REPLACE INTO session_metadata
                   (session_id, tool, start_time, end_time, duration_minutes,
                    turn_count, user_turn_count, tool_use_count,
                    total_input_tokens, total_output_tokens,
                    session_type, workspace_path, branch, model)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    session.session_id,
                    session.tool.value,
                    session.start_time.isoformat() if session.start_time else None,
                    session.end_time.isoformat() if session.end_time else None,
                    session.duration_minutes,
                    session.turn_count,
                    session.user_turn_count,
                    session.tool_use_count,
                    session.total_input_tokens,
                    session.total_output_tokens,
                    session.session_type.value,
                    session.workspace_path,
                    session.branch,
                    session.model,
                ),
            )
            return cursor.lastrowid

    def get_session_count(self, tool: Optional[str] = None,
                          since: Optional[datetime] = None) -> int:
        """Count sessions, optionally filtered."""
        query = "SELECT COUNT(*) FROM session_metadata WHERE 1=1"
        params = []
        if tool:
            query += " AND tool = ?"
            params.append(tool)
        if since:
            query += " AND start_time >= ?"
            params.append(since.isoformat())
        row = self.conn.execute(query, params).fetchone()
        return row[0] if row else 0

    # =========================================================================
    # Analysis Results
    # =========================================================================

    def save_analysis_result(self, result, tool: str) -> int:
        """Save an analysis result."""
        with self.conn:
            cursor = self.conn.execute(
                """INSERT INTO analysis_results
                   (analyzer_key, score, metrics, session_count,
                    period_start, period_end, tool)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    result.analyzer_key,
                    result.score,
                    json.dumps(result.metrics),
                    result.session_count,
                    result.period_start.isoformat() if result.period_start else "",
                    result.period_end.isoformat() if result.period_end else "",
                    tool,
                ),
            )
            return cursor.lastrowid

    def get_latest_scores(self, tool: str, limit: int = 10) -> List[Dict]:
        """Get most recent analysis scores grouped by analyzer."""
        rows = self.conn.execute(
            """SELECT analyzer_key, score, metrics, period_start, period_end, created_at
               FROM analysis_results
               WHERE tool = ?
               ORDER BY created_at DESC
               LIMIT ?""",
            (tool, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_score_history(self, analyzer_key: str, tool: str,
                          days: int = 90) -> List[Dict]:
        """Get score history for a specific analyzer."""
        since = (datetime.utcnow() - timedelta(days=days)).isoformat()
        rows = self.conn.execute(
            """SELECT score, period_start, period_end, created_at
               FROM analysis_results
               WHERE analyzer_key = ? AND tool = ? AND created_at >= ?
               ORDER BY created_at ASC""",
            (analyzer_key, tool, since),
        ).fetchall()
        return [dict(r) for r in rows]

    # =========================================================================
    # Insights
    # =========================================================================

    def save_insight(self, insight, tool: str,
                     period_start: Optional[datetime] = None,
                     period_end: Optional[datetime] = None) -> int:
        """Save a coaching insight."""
        with self.conn:
            cursor = self.conn.execute(
                """INSERT INTO insights
                   (category, title, severity, recommendation, evidence,
                    metric_key, metric_value, trend, tool,
                    period_start, period_end)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    insight.category.value,
                    insight.title,
                    insight.severity.value,
                    insight.recommendation,
                    insight.evidence,
                    insight.metric_key,
                    insight.metric_value,
                    insight.trend.value,
                    tool,
                    period_start.isoformat() if period_start else None,
                    period_end.isoformat() if period_end else None,
                ),
            )
            return cursor.lastrowid

    def get_recent_insights(self, tool: str, limit: int = 20,
                            severity: Optional[str] = None) -> List[Dict]:
        """Get recent insights, optionally filtered by severity."""
        query = "SELECT * FROM insights WHERE tool = ?"
        params: List[Any] = [tool]
        if severity:
            query += " AND severity = ?"
            params.append(severity)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        rows = self.conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    # =========================================================================
    # Reports
    # =========================================================================

    def save_report(self, report) -> int:
        """Save a complete report."""
        with self.conn:
            cursor = self.conn.execute(
                """INSERT INTO reports
                   (tool, period_start, period_end, overall_score,
                    session_count, total_turns, total_tokens,
                    total_duration_minutes, report_data)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    report.tool.value,
                    report.period_start.isoformat(),
                    report.period_end.isoformat(),
                    report.overall_score,
                    report.session_count,
                    report.total_turns,
                    report.total_tokens,
                    report.total_duration_minutes,
                    json.dumps(report.to_dict()),
                ),
            )
            return cursor.lastrowid

    def get_latest_report(self, tool: str) -> Optional[Dict]:
        """Get the most recent report for a tool."""
        row = self.conn.execute(
            """SELECT * FROM reports WHERE tool = ?
               ORDER BY created_at DESC LIMIT 1""",
            (tool,),
        ).fetchone()
        return dict(row) if row else None

    def get_report_history(self, tool: str, limit: int = 12) -> List[Dict]:
        """Get report history for trend comparison."""
        rows = self.conn.execute(
            """SELECT id, tool, period_start, period_end, overall_score,
                      session_count, total_turns, created_at
               FROM reports WHERE tool = ?
               ORDER BY period_end DESC LIMIT ?""",
            (tool, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    # =========================================================================
    # Trends
    # =========================================================================

    def save_trend_point(self, metric_key: str, value: float,
                         tool: str, measured_at: datetime,
                         period_type: str = "daily"):
        """Save a single trend data point."""
        with self.conn:
            self.conn.execute(
                """INSERT INTO trends
                   (metric_key, metric_value, tool, measured_at, period_type)
                   VALUES (?, ?, ?, ?, ?)""",
                (metric_key, value, tool, measured_at.isoformat(), period_type),
            )

    def get_trend(self, metric_key: str, tool: str,
                  days: int = 30) -> List[Dict]:
        """Get trend data for a metric."""
        since = (datetime.utcnow() - timedelta(days=days)).isoformat()
        rows = self.conn.execute(
            """SELECT metric_value, measured_at, period_type
               FROM trends
               WHERE metric_key = ? AND tool = ? AND measured_at >= ?
               ORDER BY measured_at ASC""",
            (metric_key, tool, since),
        ).fetchall()
        return [dict(r) for r in rows]

    # =========================================================================
    # Config
    # =========================================================================

    def get_config(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get a config value."""
        row = self.conn.execute(
            "SELECT value FROM config WHERE key = ?", (key,)
        ).fetchone()
        return row[0] if row else default

    def set_config(self, key: str, value: str):
        """Set a config value."""
        with self.conn:
            self.conn.execute(
                """INSERT OR REPLACE INTO config (key, value, updated_at)
                   VALUES (?, ?, datetime('now'))""",
                (key, value),
            )

    # =========================================================================
    # Maintenance
    # =========================================================================

    def cleanup_old_data(self, retention_days: int = RETENTION_DAYS):
        """Remove data older than retention period."""
        cutoff = (datetime.utcnow() - timedelta(days=retention_days)).isoformat()
        with self.conn:
            for table, col in [
                ("session_metadata", "created_at"),
                ("analysis_results", "created_at"),
                ("insights", "created_at"),
                ("trends", "measured_at"),
            ]:
                self.conn.execute(
                    f"DELETE FROM {table} WHERE {col} < ?", (cutoff,)
                )
        logger.info("Cleaned up data older than %d days", retention_days)
