"""
CUE Utilities Tests.

Tests for logger, helper functions, and utility classes.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest


# ─────────────────────────────────────────────────────────────────────────────
# CUE LOGGER TESTS
# ─────────────────────────────────────────────────────────────────────────────


class TestCueLogger:
    """Tests for CueLogger class."""

    def test_singleton_pattern(self) -> None:
        """Test that CueLogger returns same instance for same name."""
        from cue.utils import CueLogger
        
        # Clear instances for clean test
        CueLogger._instances.clear()
        
        logger1 = CueLogger("test.module")
        logger2 = CueLogger("test.module")
        
        assert logger1 is logger2

    def test_different_names_different_instances(self) -> None:
        """Test that different names get different instances."""
        from cue.utils import CueLogger
        
        CueLogger._instances.clear()
        
        logger1 = CueLogger("module.one")
        logger2 = CueLogger("module.two")
        
        assert logger1 is not logger2

    def test_logger_initialization(self) -> None:
        """Test that logger initializes correctly."""
        from cue.utils import CueLogger
        
        CueLogger._instances.clear()
        
        logger = CueLogger("test.init")
        
        assert logger.name == "test.init"
        assert logger.logger is not None
        assert hasattr(logger, "_initialized")

    def test_setup_logging_basic(self) -> None:
        """Test setup_logging with basic configuration."""
        from cue.utils import CueLogger
        
        # This should not raise
        CueLogger.setup_logging(
            level="DEBUG",
            show_timestamps=False,
            rich_tracebacks=False,
        )
        
        root_logger = logging.getLogger()
        assert root_logger.level == logging.DEBUG

    def test_setup_logging_with_file(self, tmp_path: Path) -> None:
        """Test setup_logging with file handler."""
        from cue.utils import CueLogger
        
        log_file = tmp_path / "test.log"
        
        CueLogger.setup_logging(
            level="INFO",
            log_to_file=True,
            log_file_path=str(log_file),
        )
        
        # The file handler should be added
        root_logger = logging.getLogger()
        file_handlers = [h for h in root_logger.handlers if isinstance(h, logging.FileHandler)]
        assert len(file_handlers) >= 1

    def test_debug_message(self) -> None:
        """Test debug logging."""
        from cue.utils import CueLogger
        
        CueLogger._instances.clear()
        logger = CueLogger("test.debug")
        
        with patch.object(logger.logger, "debug") as mock_debug:
            logger.debug("Test debug message")
            mock_debug.assert_called_once_with("Test debug message")

    def test_info_message(self) -> None:
        """Test info logging."""
        from cue.utils import CueLogger
        
        CueLogger._instances.clear()
        logger = CueLogger("test.info")
        
        with patch.object(logger.logger, "info") as mock_info:
            logger.info("Test info message")
            mock_info.assert_called_once_with("Test info message")

    def test_warning_message(self) -> None:
        """Test warning logging."""
        from cue.utils import CueLogger
        
        CueLogger._instances.clear()
        logger = CueLogger("test.warning")
        
        with patch.object(logger.logger, "warning") as mock_warning:
            logger.warning("Test warning message")
            mock_warning.assert_called_once_with("Test warning message")

    def test_error_message(self) -> None:
        """Test error logging."""
        from cue.utils import CueLogger
        
        CueLogger._instances.clear()
        logger = CueLogger("test.error")
        
        with patch.object(logger.logger, "error") as mock_error:
            logger.error("Test error message")
            mock_error.assert_called_once_with("Test error message")

    def test_exception_message(self) -> None:
        """Test exception logging."""
        from cue.utils import CueLogger
        
        CueLogger._instances.clear()
        logger = CueLogger("test.exception")
        
        with patch.object(logger.logger, "exception") as mock_exception:
            logger.exception("Test exception message")
            mock_exception.assert_called_once_with("Test exception message")

    def test_success_message(self) -> None:
        """Test success logging with custom styling."""
        from cue.utils import CueLogger
        
        CueLogger._instances.clear()
        logger = CueLogger("test.success")
        
        with patch.object(logger.logger, "info") as mock_info:
            logger.success("Test success message")
            # Should include the checkmark and styling
            mock_info.assert_called_once()
            call_args = mock_info.call_args[0][0]
            assert "✔" in call_args
            assert "Test success message" in call_args

    def test_status_message(self) -> None:
        """Test status logging with custom styling."""
        from cue.utils import CueLogger
        
        CueLogger._instances.clear()
        logger = CueLogger("test.status")
        
        with patch.object(logger.logger, "info") as mock_info:
            logger.status("Test status message")
            # Should include the arrow and styling
            mock_info.assert_called_once()
            call_args = mock_info.call_args[0][0]
            assert "→" in call_args
            assert "Test status message" in call_args


class TestGetLogger:
    """Tests for get_logger function."""

    def test_get_logger_returns_cue_logger(self) -> None:
        """Test that get_logger returns a CueLogger instance."""
        from cue.utils import get_logger, CueLogger
        
        logger = get_logger("test.get")
        
        assert isinstance(logger, CueLogger)


# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTION TESTS
# ─────────────────────────────────────────────────────────────────────────────


class TestHelperFunctions:
    """Tests for utility helper functions."""

    def test_get_timestamp(self) -> None:
        """Test get_timestamp returns ISO format."""
        from cue.utils import get_timestamp
        
        timestamp = get_timestamp()
        
        # Should be parseable as ISO format
        parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        assert parsed.tzinfo is not None

    def test_get_timestamp_id(self) -> None:
        """Test get_timestamp_id returns filename-safe format."""
        from cue.utils import get_timestamp_id
        
        timestamp_id = get_timestamp_id()
        
        # Should match format YYYYMMDD_HHMMSS
        assert "_" in timestamp_id
        assert len(timestamp_id) == 15  # 8 + 1 + 6

    def test_ensure_path_directory(self, tmp_path: Path) -> None:
        """Test ensure_path creates directory."""
        from cue.utils import ensure_path
        
        new_dir = tmp_path / "new_directory"
        assert not new_dir.exists()
        
        result = ensure_path(new_dir)
        
        assert new_dir.exists()
        assert new_dir.is_dir()
        assert result == new_dir

    def test_ensure_path_nested_directory(self, tmp_path: Path) -> None:
        """Test ensure_path creates nested directories."""
        from cue.utils import ensure_path
        
        nested_dir = tmp_path / "a" / "b" / "c"
        assert not nested_dir.exists()
        
        ensure_path(nested_dir)
        
        assert nested_dir.exists()

    def test_ensure_path_file(self, tmp_path: Path) -> None:
        """Test ensure_path creates parent for file path."""
        from cue.utils import ensure_path
        
        file_path = tmp_path / "subdir" / "file.txt"
        assert not file_path.parent.exists()
        
        result = ensure_path(file_path)
        
        assert file_path.parent.exists()
        assert result == file_path

    def test_ensure_path_string(self, tmp_path: Path) -> None:
        """Test ensure_path works with string paths."""
        from cue.utils import ensure_path
        
        new_dir = str(tmp_path / "string_dir")
        
        result = ensure_path(new_dir)
        
        assert Path(new_dir).exists()
        assert isinstance(result, Path)

    def test_format_duration_seconds(self) -> None:
        """Test format_duration for durations under 60 seconds."""
        from cue.utils import format_duration
        
        assert format_duration(0.5) == "0.5s"
        assert format_duration(5) == "5.0s"
        assert format_duration(45.5) == "45.5s"
        assert format_duration(59.9) == "59.9s"

    def test_format_duration_minutes(self) -> None:
        """Test format_duration for durations under 60 minutes."""
        from cue.utils import format_duration
        
        assert format_duration(60) == "1m 0s"
        assert format_duration(90) == "1m 30s"
        assert format_duration(125) == "2m 5s"
        assert format_duration(3599) == "59m 59s"

    def test_format_duration_hours(self) -> None:
        """Test format_duration for durations over 60 minutes."""
        from cue.utils import format_duration
        
        assert format_duration(3600) == "1h 0m"
        assert format_duration(3660) == "1h 1m"
        assert format_duration(7200) == "2h 0m"
        assert format_duration(5400) == "1h 30m"

    def test_truncate_text_short(self) -> None:
        """Test truncate_text with text shorter than max."""
        from cue.utils import truncate_text
        
        result = truncate_text("Hello", max_length=50)
        
        assert result == "Hello"

    def test_truncate_text_exact(self) -> None:
        """Test truncate_text with text exactly at max."""
        from cue.utils import truncate_text
        
        result = truncate_text("12345", max_length=5)
        
        assert result == "12345"

    def test_truncate_text_long(self) -> None:
        """Test truncate_text with text longer than max."""
        from cue.utils import truncate_text
        
        result = truncate_text("Hello World!", max_length=8)
        
        assert result == "Hello..."
        assert len(result) == 8

    def test_truncate_text_custom_suffix(self) -> None:
        """Test truncate_text with custom suffix."""
        from cue.utils import truncate_text
        
        result = truncate_text("Hello World!", max_length=10, suffix=">>")
        
        assert result == "Hello Wo>>"
        assert len(result) == 10

    def test_safe_call_success(self) -> None:
        """Test safe_call with successful function."""
        from cue.utils import safe_call
        
        def add(a, b):
            return a + b
        
        result = safe_call(add, 2, 3)
        
        assert result == 5

    def test_safe_call_exception(self) -> None:
        """Test safe_call catches exceptions."""
        from cue.utils import safe_call
        
        def failing_func():
            raise ValueError("Error!")
        
        result = safe_call(failing_func, default="fallback")
        
        assert result == "fallback"

    def test_safe_call_exception_default_none(self) -> None:
        """Test safe_call returns None by default on exception."""
        from cue.utils import safe_call
        
        def failing_func():
            raise ValueError("Error!")
        
        result = safe_call(failing_func)
        
        assert result is None

    def test_safe_call_with_logger(self) -> None:
        """Test safe_call logs error when logger provided."""
        from cue.utils import safe_call, CueLogger
        
        CueLogger._instances.clear()
        mock_logger = MagicMock()
        
        def failing_func():
            raise ValueError("Test error")
        
        safe_call(failing_func, logger=mock_logger)
        
        mock_logger.error.assert_called_once()

    def test_safe_call_with_kwargs(self) -> None:
        """Test safe_call passes kwargs correctly."""
        from cue.utils import safe_call
        
        def greet(name, greeting="Hello"):
            return f"{greeting}, {name}!"
        
        result = safe_call(greet, "World", greeting="Hi")
        
        assert result == "Hi, World!"


class TestPrintFunctions:
    """Tests for print utility functions."""

    def test_print_banner(self) -> None:
        """Test print_banner outputs banner."""
        from cue.utils import print_banner
        
        with patch("cue.utils.output_console") as mock_console:
            print_banner()
            
            mock_console.print.assert_called_once()

    def test_print_status_panel(self) -> None:
        """Test print_status_panel outputs panel."""
        from cue.utils import print_status_panel
        
        with patch("cue.utils.output_console") as mock_console:
            print_status_panel(
                state="IDLE",
                description="System ready",
                color="#00FF00",
            )
            
            mock_console.print.assert_called_once()


class TestConsoleInstances:
    """Tests for global console instances."""

    def test_console_exists(self) -> None:
        """Test that console instance exists."""
        from cue.utils import console
        
        assert console is not None

    def test_output_console_exists(self) -> None:
        """Test that output_console instance exists."""
        from cue.utils import output_console
        
        assert output_console is not None

    def test_console_has_theme(self) -> None:
        """Test that console has CUE theme."""
        from cue.utils import console, CUE_THEME
        
        # Console should be configured with theme
        assert console is not None
