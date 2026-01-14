"""
CUE Configuration Module.

This module provides a unified configuration system that merges:
- Environment variables (.env) for secrets
- YAML configuration (config.yaml) for behavioral settings

Uses Pydantic for validation and type safety. The configuration is loaded
once at import time and available as a singleton via `from cue.config import settings`.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# ─────────────────────────────────────────────────────────────────────────────
# NESTED CONFIGURATION MODELS
# ─────────────────────────────────────────────────────────────────────────────


class AudioConfig(BaseModel):
    """Audio capture configuration."""

    sample_rate: int = Field(default=16000, ge=8000, le=48000)
    channels: int = Field(default=1, ge=1, le=2)
    dtype: str = Field(default="int16")
    chunk_duration_ms: int = Field(default=100, ge=10, le=1000)
    max_recording_seconds: int = Field(default=60, ge=5, le=300)

    @field_validator("dtype")
    @classmethod
    def validate_dtype(cls, v: str) -> str:
        """Validate audio dtype is supported."""
        allowed = {"int16", "int32", "float32"}
        if v not in allowed:
            raise ValueError(f"dtype must be one of {allowed}")
        return v


class HardwareConfig(BaseModel):
    """Hardware simulation configuration."""

    audio: AudioConfig = Field(default_factory=AudioConfig)
    hotkey: str = Field(default="space")
    debounce_ms: int = Field(default=200, ge=50, le=1000)
    input_mode: Literal["toggle", "hold"] = Field(default="toggle")
    double_tap_ms: int = Field(default=300, ge=100, le=1000)
    min_hold_duration_ms: int = Field(default=500, ge=100, le=2000)


class LEDStateConfig(BaseModel):
    """Configuration for a single LED state."""

    color: str = Field(pattern=r"^#[0-9A-Fa-f]{6}$")
    label: str
    description: str = ""
    blink: bool = False
    blink_interval_ms: int = Field(default=500, ge=100, le=2000)
    animate: bool = False


class LEDStatesConfig(BaseModel):
    """All LED state configurations."""

    idle: LEDStateConfig
    recording: LEDStateConfig
    processing: LEDStateConfig
    error: LEDStateConfig
    success: LEDStateConfig


class LLMConfig(BaseModel):
    """LLM (Ollama) configuration."""

    model: str = Field(default="llama3.1")
    host: str = Field(default="http://localhost:11434")
    timeout_seconds: int = Field(default=30, ge=5, le=120)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, ge=128, le=8192)
    scratchpad_prompt: str
    completion_test_prompt: str
    extraction_prompt: str


class DeepgramConfig(BaseModel):
    """Deepgram STT configuration."""

    model: str = Field(default="nova-2")
    language: str = Field(default="en-US")
    smart_format: bool = True
    punctuate: bool = True
    diarize: bool = False
    utterances: bool = False


class PathsConfig(BaseModel):
    """File path configuration."""

    tickets_dir: str = Field(default="tickets")
    temp_audio_dir: str = Field(default=".temp_audio")
    ticket_template: str

    @field_validator("tickets_dir", "temp_audio_dir")
    @classmethod
    def validate_path(cls, v: str) -> str:
        """Ensure path doesn't contain dangerous patterns."""
        if ".." in v or v.startswith("/"):
            raise ValueError("Path must be relative and not contain '..'")
        return v


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    show_timestamps: bool = True
    rich_tracebacks: bool = True
    log_to_file: bool = False
    log_file_path: str = "cue.log"


class UIConfig(BaseModel):
    """UI theme configuration."""

    title: str = Field(default="CUE v0.1 • Intent Capture System")
    border_style: str = Field(default="rounded")
    primary_color: str = Field(default="#00D4FF", pattern=r"^#[0-9A-Fa-f]{6}$")
    secondary_color: str = Field(default="#FF6B6B", pattern=r"^#[0-9A-Fa-f]{6}$")
    accent_color: str = Field(default="#FFE66D", pattern=r"^#[0-9A-Fa-f]{6}$")
    show_audio_meter: bool = True
    refresh_rate_hz: int = Field(default=10, ge=1, le=60)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN SETTINGS CLASS
# ─────────────────────────────────────────────────────────────────────────────


def _find_project_root() -> Path:
    """Find the project root directory by looking for config.yaml."""
    current = Path.cwd()
    
    # Check current directory and parents
    for parent in [current, *current.parents]:
        if (parent / "config.yaml").exists():
            return parent
        if (parent / "pyproject.toml").exists():
            return parent
    
    return current


def _load_yaml_config() -> dict[str, Any]:
    """Load configuration from YAML file."""
    project_root = _find_project_root()
    config_path = project_root / "config.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}\n"
            "Please ensure config.yaml exists in the project root."
        )
    
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    if config is None:
        raise ValueError("config.yaml is empty or invalid")
    
    return config


class Settings(BaseSettings):
    """
    Main application settings.
    
    Merges environment variables (secrets) with YAML configuration (behavior).
    Validates all settings at load time for fail-fast behavior.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Secrets from .env
    deepgram_api_key: str = Field(
        ...,
        description="Deepgram API key for speech-to-text",
        min_length=10,
    )
    ollama_host: str | None = Field(
        default=None,
        description="Override Ollama host from environment",
    )

    # Behavioral configuration from config.yaml
    hardware: HardwareConfig
    led_states: LEDStatesConfig
    llm: LLMConfig
    deepgram: DeepgramConfig
    paths: PathsConfig
    logging: LoggingConfig
    ui: UIConfig

    # Computed paths
    project_root: Path = Field(default_factory=_find_project_root)

    def __init__(self, **kwargs: Any) -> None:
        """Initialize settings by merging YAML config with env vars."""
        yaml_config = _load_yaml_config()
        
        # Merge YAML config into kwargs (env vars take precedence)
        for key, value in yaml_config.items():
            if key not in kwargs:
                kwargs[key] = value
        
        super().__init__(**kwargs)
        
        # Override LLM host if set in environment
        if self.ollama_host:
            self.llm.host = self.ollama_host

    @property
    def tickets_path(self) -> Path:
        """Get absolute path to tickets directory."""
        return self.project_root / self.paths.tickets_dir

    @property
    def temp_audio_path(self) -> Path:
        """Get absolute path to temporary audio directory."""
        return self.project_root / self.paths.temp_audio_dir

    def ensure_directories(self) -> None:
        """Create required directories if they don't exist."""
        self.tickets_path.mkdir(parents=True, exist_ok=True)
        self.temp_audio_path.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Get the singleton Settings instance.
    
    Uses lru_cache to ensure settings are only loaded once.
    
    Returns:
        The application Settings instance.
        
    Raises:
        ValidationError: If configuration is invalid.
        FileNotFoundError: If config.yaml is missing.
    """
    return Settings()


# Singleton instance for easy importing
# Usage: from cue.config import settings
settings = get_settings()
