"""
CUE Configuration Tests.

Tests for configuration loading, validation, and fail-fast behavior.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

import pytest
from pydantic import ValidationError

if TYPE_CHECKING:
    from cue.config import Settings


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG LOADING TESTS
# ─────────────────────────────────────────────────────────────────────────────


class TestConfigLoading:
    """Tests for configuration loading."""

    def test_config_loads_from_yaml(self, temp_project_dir: Path) -> None:
        """Test that configuration loads successfully from YAML."""
        # Clear cache and reimport
        from cue.config import get_settings
        get_settings.cache_clear()
        
        with patch.dict(os.environ, {"DEEPGRAM_API_KEY": "test_key_12345"}):
            settings = get_settings()
        
        assert settings.hardware.audio.sample_rate == 16000
        assert settings.hardware.hotkey == "space"
        assert settings.llm.model == "llama3.1"

    def test_config_validates_audio_sample_rate(self, temp_project_dir: Path) -> None:
        """Test that audio sample rate is validated."""
        from cue.config import AudioConfig
        
        # Valid sample rate
        config = AudioConfig(sample_rate=16000)
        assert config.sample_rate == 16000
        
        # Invalid sample rate (too low)
        with pytest.raises(ValidationError):
            AudioConfig(sample_rate=1000)
        
        # Invalid sample rate (too high)
        with pytest.raises(ValidationError):
            AudioConfig(sample_rate=100000)

    def test_config_validates_led_color(self, temp_project_dir: Path) -> None:
        """Test that LED colors are validated as hex codes."""
        from cue.config import LEDStateConfig
        
        # Valid hex color
        config = LEDStateConfig(color="#FF0000", label="Test")
        assert config.color == "#FF0000"
        
        # Invalid color format
        with pytest.raises(ValidationError):
            LEDStateConfig(color="red", label="Test")
        
        # Invalid hex length
        with pytest.raises(ValidationError):
            LEDStateConfig(color="#FFF", label="Test")

    def test_config_validates_dtype(self, temp_project_dir: Path) -> None:
        """Test that audio dtype is validated."""
        from cue.config import AudioConfig
        
        # Valid dtypes
        for dtype in ["int16", "int32", "float32"]:
            config = AudioConfig(dtype=dtype)
            assert config.dtype == dtype
        
        # Invalid dtype
        with pytest.raises(ValidationError):
            AudioConfig(dtype="int8")

    def test_config_validates_paths(self, temp_project_dir: Path) -> None:
        """Test that paths are validated to be relative."""
        from cue.config import PathsConfig
        
        # Valid relative path
        config = PathsConfig(
            tickets_dir="tickets",
            temp_audio_dir=".temp",
            ticket_template="Template",
        )
        assert config.tickets_dir == "tickets"
        
        # Invalid absolute path
        with pytest.raises(ValidationError):
            PathsConfig(
                tickets_dir="/absolute/path",
                temp_audio_dir=".temp",
                ticket_template="Template",
            )
        
        # Invalid path with parent traversal
        with pytest.raises(ValidationError):
            PathsConfig(
                tickets_dir="../parent",
                temp_audio_dir=".temp",
                ticket_template="Template",
            )


# ─────────────────────────────────────────────────────────────────────────────
# FAIL-FAST TESTS
# ─────────────────────────────────────────────────────────────────────────────


class TestFailFast:
    """Tests for fail-fast validation behavior."""

    def test_missing_api_key_fails(self, temp_project_dir: Path) -> None:
        """Test that missing API key causes immediate failure."""
        from cue.config import get_settings
        get_settings.cache_clear()
        
        # Remove API key from environment
        env_file = temp_project_dir / ".env"
        env_file.write_text("")
        
        with patch.dict(os.environ, {}, clear=True):
            # Remove any existing DEEPGRAM_API_KEY from env
            os.environ.pop("DEEPGRAM_API_KEY", None)
            
            with pytest.raises(ValidationError) as exc_info:
                get_settings()
            
            assert "deepgram_api_key" in str(exc_info.value).lower()

    def test_missing_config_file_fails(self, tmp_path: Path) -> None:
        """Test that missing config.yaml causes immediate failure."""
        from cue.config import get_settings, _find_project_root
        get_settings.cache_clear()
        
        original_cwd = os.getcwd()
        os.chdir(tmp_path)
        
        try:
            # No config.yaml exists
            with pytest.raises(FileNotFoundError) as exc_info:
                get_settings()
            
            assert "config.yaml" in str(exc_info.value)
        finally:
            os.chdir(original_cwd)

    def test_invalid_log_level_fails(self, temp_project_dir: Path) -> None:
        """Test that invalid log level is rejected."""
        from cue.config import LoggingConfig
        
        # Valid levels
        for level in ["DEBUG", "INFO", "WARNING", "ERROR"]:
            config = LoggingConfig(level=level)
            assert config.level == level
        
        # Invalid level
        with pytest.raises(ValidationError):
            LoggingConfig(level="TRACE")


# ─────────────────────────────────────────────────────────────────────────────
# SINGLETON TESTS
# ─────────────────────────────────────────────────────────────────────────────


class TestSingleton:
    """Tests for singleton pattern."""

    def test_settings_is_singleton(self, temp_project_dir: Path) -> None:
        """Test that settings returns the same instance."""
        from cue.config import get_settings
        get_settings.cache_clear()
        
        with patch.dict(os.environ, {"DEEPGRAM_API_KEY": "test_key_12345"}):
            settings1 = get_settings()
            settings2 = get_settings()
        
        assert settings1 is settings2

    def test_settings_properties(self, temp_project_dir: Path) -> None:
        """Test computed properties."""
        from cue.config import get_settings
        get_settings.cache_clear()
        
        with patch.dict(os.environ, {"DEEPGRAM_API_KEY": "test_key_12345"}):
            settings = get_settings()
        
        assert settings.tickets_path == settings.project_root / "tickets"
        assert settings.temp_audio_path == settings.project_root / ".temp_audio"
