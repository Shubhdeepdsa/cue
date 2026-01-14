"""
CUE Utilities Module.

Provides custom logging, helper functions, and common utilities used
throughout the application. All user-facing output uses Rich console.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, TypeVar

from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.text import Text
from rich.theme import Theme

if TYPE_CHECKING:
    from collections.abc import Callable

# ─────────────────────────────────────────────────────────────────────────────
# RICH CONSOLE SINGLETON
# ─────────────────────────────────────────────────────────────────────────────

# Custom theme for CUE
CUE_THEME = Theme({
    "cue.primary": "#00D4FF",
    "cue.secondary": "#FF6B6B",
    "cue.accent": "#FFE66D",
    "cue.success": "#00FF88",
    "cue.warning": "#FFAA00",
    "cue.error": "#FF3366",
    "cue.muted": "#6B7280",
    "cue.idle": "#00FF88",
    "cue.recording": "#FF3366",
    "cue.processing": "#FFAA00",
})

# Global console instance
console = Console(theme=CUE_THEME, stderr=True)
output_console = Console(theme=CUE_THEME)


# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM LOGGER
# ─────────────────────────────────────────────────────────────────────────────


class CueLogger:
    """
    Custom logger that wraps Python's logging with Rich formatting.
    
    Provides consistent, beautiful logging output across the application.
    All output goes through Rich for proper terminal handling.
    
    Attributes:
        name: The logger name (module name typically).
        logger: The underlying Python logger instance.
    """

    _instances: dict[str, "CueLogger"] = {}

    def __new__(cls, name: str) -> "CueLogger":
        """Ensure singleton per logger name."""
        if name not in cls._instances:
            instance = super().__new__(cls)
            cls._instances[name] = instance
        return cls._instances[name]

    def __init__(self, name: str) -> None:
        """
        Initialize the CUE logger.
        
        Args:
            name: Logger name, typically __name__ of the calling module.
        """
        if hasattr(self, "_initialized"):
            return
        
        self.name = name
        self.logger = logging.getLogger(name)
        self._initialized = True

    @classmethod
    def setup_logging(
        cls,
        level: str = "INFO",
        show_timestamps: bool = True,
        rich_tracebacks: bool = True,
        log_to_file: bool = False,
        log_file_path: str = "cue.log",
    ) -> None:
        """
        Configure the root logger with Rich formatting.
        
        Should be called once at application startup.
        
        Args:
            level: Log level string (DEBUG, INFO, WARNING, ERROR).
            show_timestamps: Whether to show timestamps in log output.
            rich_tracebacks: Whether to use Rich for exception formatting.
            log_to_file: Whether to also log to a file.
            log_file_path: Path to the log file.
        """
        log_level = getattr(logging, level.upper(), logging.INFO)
        
        # Configure Rich handler for console
        rich_handler = RichHandler(
            console=console,
            show_time=show_timestamps,
            show_path=False,
            rich_tracebacks=rich_tracebacks,
            tracebacks_show_locals=True,
            markup=True,
        )
        rich_handler.setLevel(log_level)
        
        handlers: list[logging.Handler] = [rich_handler]
        
        # Optionally add file handler
        if log_to_file:
            file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
            file_handler.setLevel(log_level)
            file_formatter = logging.Formatter(
                "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
            )
            file_handler.setFormatter(file_formatter)
            handlers.append(file_handler)
        
        # Configure root logger
        logging.basicConfig(
            level=log_level,
            handlers=handlers,
            force=True,
        )
        
        # Suppress noisy third-party loggers
        for noisy_logger in ["httpx", "httpcore", "urllib3", "asyncio"]:
            logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    def debug(self, message: str, **kwargs: Any) -> None:
        """Log a debug message."""
        self.logger.debug(message, **kwargs)

    def info(self, message: str, **kwargs: Any) -> None:
        """Log an info message."""
        self.logger.info(message, **kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        """Log a warning message."""
        self.logger.warning(message, **kwargs)

    def error(self, message: str, **kwargs: Any) -> None:
        """Log an error message."""
        self.logger.error(message, **kwargs)

    def exception(self, message: str, **kwargs: Any) -> None:
        """Log an exception with traceback."""
        self.logger.exception(message, **kwargs)

    def success(self, message: str) -> None:
        """Log a success message with green styling."""
        self.logger.info(f"[cue.success]✔ {message}[/cue.success]")

    def status(self, message: str) -> None:
        """Log a status update with accent styling."""
        self.logger.info(f"[cue.accent]→ {message}[/cue.accent]")


def get_logger(name: str) -> CueLogger:
    """
    Get a CueLogger instance for the given name.
    
    Args:
        name: Logger name, typically __name__.
        
    Returns:
        A CueLogger instance.
    """
    return CueLogger(name)


# ─────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────


def get_timestamp() -> str:
    """
    Get current UTC timestamp in ISO format.
    
    Returns:
        ISO-formatted timestamp string.
    """
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def get_timestamp_id() -> str:
    """
    Get a timestamp-based ID suitable for filenames.
    
    Returns:
        Timestamp ID in format YYYYMMDD_HHMMSS.
    """
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def ensure_path(path: Path | str) -> Path:
    """
    Ensure a path exists, creating directories if needed.
    
    Args:
        path: The path to ensure exists.
        
    Returns:
        The Path object.
    """
    path = Path(path)
    if path.suffix:
        # It's a file, ensure parent exists
        path.parent.mkdir(parents=True, exist_ok=True)
    else:
        # It's a directory
        path.mkdir(parents=True, exist_ok=True)
    return path


def format_duration(seconds: float) -> str:
    """
    Format a duration in seconds to a human-readable string.
    
    Args:
        seconds: Duration in seconds.
        
    Returns:
        Formatted duration string (e.g., "1m 30s" or "45s").
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    
    minutes = int(seconds // 60)
    remaining_seconds = seconds % 60
    
    if minutes < 60:
        return f"{minutes}m {remaining_seconds:.0f}s"
    
    hours = int(minutes // 60)
    remaining_minutes = minutes % 60
    return f"{hours}h {remaining_minutes}m"


def truncate_text(text: str, max_length: int = 50, suffix: str = "...") -> str:
    """
    Truncate text to a maximum length with suffix.
    
    Args:
        text: The text to truncate.
        max_length: Maximum length including suffix.
        suffix: Suffix to append when truncating.
        
    Returns:
        Truncated text.
    """
    if len(text) <= max_length:
        return text
    return text[: max_length - len(suffix)] + suffix


T = TypeVar("T")


def safe_call(
    func: Callable[..., T],
    *args: Any,
    default: T | None = None,
    logger: CueLogger | None = None,
    **kwargs: Any,
) -> T | None:
    """
    Safely call a function, catching and logging exceptions.
    
    Args:
        func: The function to call.
        *args: Positional arguments for the function.
        default: Default value to return on exception.
        logger: Logger for error reporting.
        **kwargs: Keyword arguments for the function.
        
    Returns:
        Function result or default value on exception.
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if logger:
            logger.error(f"Error in {func.__name__}: {e}")
        return default


def print_banner() -> None:
    """Print the CUE startup banner."""
    banner_text = Text()
    banner_text.append("╔═══════════════════════════════════════════════════════════╗\n", style="cue.primary")
    banner_text.append("║", style="cue.primary")
    banner_text.append("           ", style="")
    banner_text.append("CUE", style="bold cue.primary")
    banner_text.append(" v0.1 • Intent Capture System", style="cue.accent")
    banner_text.append("           ", style="")
    banner_text.append("║\n", style="cue.primary")
    banner_text.append("║", style="cue.primary")
    banner_text.append("              Local Simulation Edition              ", style="cue.muted")
    banner_text.append("║\n", style="cue.primary")
    banner_text.append("╚═══════════════════════════════════════════════════════════╝", style="cue.primary")
    
    output_console.print(banner_text)


def print_status_panel(
    state: str,
    description: str,
    color: str,
) -> None:
    """
    Print a status panel with the current system state.
    
    Args:
        state: The current state label.
        description: Description of the state.
        color: Hex color code for the panel border.
    """
    panel = Panel(
        Text(description, style=f"bold {color}"),
        title=f"[bold]{state}[/bold]",
        border_style=color,
        padding=(0, 2),
    )
    output_console.print(panel)
