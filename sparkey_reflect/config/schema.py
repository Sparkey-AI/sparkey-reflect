"""
Configuration Schema

Pydantic models for validating Reflect configuration.
"""

from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field

from sparkey_reflect.config.defaults import (
    DEFAULT_ANALYSIS_DAYS,
    DEFAULT_MODEL,
    DEFAULT_PRESET,
    MAX_OUTPUT_TOKENS,
    RETENTION_DAYS,
)


class ReflectConfig(BaseModel):
    """Top-level Reflect configuration."""

    # Analysis
    default_tool: Optional[str] = None  # auto-detect if None
    default_days: int = Field(default=DEFAULT_ANALYSIS_DAYS, ge=1, le=365)
    default_preset: str = DEFAULT_PRESET

    # Storage
    db_path: Optional[str] = None  # uses default if None
    retention_days: int = Field(default=RETENTION_DAYS, ge=30, le=730)

    # Privacy
    sync_enabled: bool = False
    api_key: Optional[str] = None
    api_url: str = "https://api.sparkey.ai"

    # Analyzers
    enabled_analyzers: Optional[List[str]] = None
    disabled_analyzers: Optional[List[str]] = None

    # LLM
    model: str = DEFAULT_MODEL
    max_output_tokens: int = MAX_OUTPUT_TOKENS
    api_key_env: str = "ANTHROPIC_API_KEY"

    # Workspace overrides
    workspace_paths: Optional[List[str]] = None

    def get_db_path(self) -> Path:
        if self.db_path:
            return Path(self.db_path)
        from sparkey_reflect.config.defaults import DEFAULT_DB_PATH
        return DEFAULT_DB_PATH
