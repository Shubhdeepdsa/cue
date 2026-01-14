"""
CUE Printer Module.

Handles the generation and output of intent tickets.
Tickets are formatted using templates defined in config.yaml
and saved to the tickets/ directory.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text

from cue.utils import get_logger, get_timestamp, get_timestamp_id, output_console

if TYPE_CHECKING:
    from cue.intelligence import IntentResult

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# TICKET PRINTER
# ─────────────────────────────────────────────────────────────────────────────


class TicketPrinter:
    """
    Generates and saves formatted intent tickets.
    
    Uses a template-based approach for consistent, aesthetic
    ticket output in both console and file formats.
    
    Attributes:
        tickets_dir: Directory path for saving tickets.
        template: Ticket template string with placeholders.
    """

    def __init__(
        self,
        tickets_dir: Path,
        template: str,
    ) -> None:
        """
        Initialize the ticket printer.
        
        Args:
            tickets_dir: Directory to save tickets.
            template: Template string with {placeholders}.
        """
        self.tickets_dir = Path(tickets_dir)
        self.template = template
        
        # Ensure tickets directory exists
        self.tickets_dir.mkdir(parents=True, exist_ok=True)
        
        # Track ticket count for IDs
        self._ticket_count = self._get_existing_ticket_count()
        
        logger.debug(f"Ticket printer initialized (dir: {tickets_dir})")

    def _get_existing_ticket_count(self) -> int:
        """Count existing tickets in the directory."""
        if not self.tickets_dir.exists():
            return 0
        return len(list(self.tickets_dir.glob("ticket_*.txt")))

    def _format_entities(self, entities: dict) -> str:
        """
        Format entities dict for display.
        
        Args:
            entities: Key-value pairs of entities.
            
        Returns:
            Formatted string representation.
        """
        if not entities:
            return "None"
        
        parts = [f"{k}: {v}" for k, v in entities.items()]
        return ", ".join(parts)

    def generate_ticket(
        self,
        intent_result: "IntentResult",
    ) -> tuple[str, Path]:
        """
        Generate a ticket from an intent result.
        
        Args:
            intent_result: The processed intent data.
            
        Returns:
            Tuple of (formatted ticket content, file path).
        """
        self._ticket_count += 1
        ticket_id = f"{self._ticket_count:05d}"
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        
        # Format ticket using template
        ticket_content = self.template.format(
            id=ticket_id,
            timestamp=timestamp,
            intent=intent_result.intent,
            action=intent_result.action,
            entities=self._format_entities(intent_result.entities),
            confidence=f"{intent_result.confidence * 100:.1f}",
            transcription=intent_result.raw_transcription,
        )
        
        # Generate filename
        file_timestamp = get_timestamp_id()
        filename = f"ticket_{file_timestamp}_{ticket_id}.txt"
        filepath = self.tickets_dir / filename
        
        logger.debug(f"Generated ticket #{ticket_id}")
        
        return ticket_content, filepath

    def save_ticket(
        self,
        intent_result: "IntentResult",
    ) -> Path:
        """
        Generate and save a ticket to disk.
        
        Args:
            intent_result: The processed intent data.
            
        Returns:
            Path to the saved ticket file.
        """
        ticket_content, filepath = self.generate_ticket(intent_result)
        
        # Save to file
        filepath.write_text(ticket_content, encoding="utf-8")
        
        logger.info(f"Ticket saved: {filepath}")
        
        return filepath

    def print_ticket(
        self,
        intent_result: "IntentResult",
        save: bool = True,
    ) -> Path | None:
        """
        Print ticket to console and optionally save to file.
        
        Args:
            intent_result: The processed intent data.
            save: Whether to also save to file.
            
        Returns:
            Path to saved file if save=True, else None.
        """
        ticket_content, filepath = self.generate_ticket(intent_result)
        
        # Print to console with styling
        output_console.print()
        output_console.print(
            Panel(
                Text(ticket_content, style="bold"),
                title="[bold green]🎫 TICKET GENERATED[/bold green]",
                border_style="green",
                padding=(1, 2),
            )
        )
        
        # Save if requested
        if save:
            filepath.write_text(ticket_content, encoding="utf-8")
            output_console.print(
                f"\n[dim]Saved to: {filepath}[/dim]",
            )
            logger.info(f"Ticket saved: {filepath}")
            return filepath
        
        return None

    def print_analysis(
        self,
        intent_result: "IntentResult",
    ) -> None:
        """
        Print the LLM analysis for debugging.
        
        Args:
            intent_result: The intent result with analysis.
        """
        if not intent_result.analysis:
            return
        
        output_console.print()
        output_console.print(
            Panel(
                Text(intent_result.analysis, style="dim"),
                title="[bold cyan]🧠 LLM Analysis[/bold cyan]",
                border_style="cyan",
                padding=(1, 2),
            )
        )


# ─────────────────────────────────────────────────────────────────────────────
# JSON EXPORTER
# ─────────────────────────────────────────────────────────────────────────────


class JSONExporter:
    """
    Exports intent results as JSON for structured data processing.
    
    Provides machine-readable output in addition to
    human-readable tickets.
    """

    def __init__(self, output_dir: Path) -> None:
        """
        Initialize the JSON exporter.
        
        Args:
            output_dir: Directory for JSON output.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export(
        self,
        intent_result: "IntentResult",
        include_analysis: bool = False,
    ) -> Path:
        """
        Export intent result as JSON.
        
        Args:
            intent_result: The intent data to export.
            include_analysis: Include full LLM analysis.
            
        Returns:
            Path to the saved JSON file.
        """
        data = {
            "timestamp": get_timestamp(),
            "intent": intent_result.intent,
            "action": intent_result.action,
            "entities": intent_result.entities,
            "confidence": intent_result.confidence,
            "transcription": intent_result.raw_transcription,
        }
        
        if include_analysis:
            data["analysis"] = intent_result.analysis
        
        filename = f"intent_{get_timestamp_id()}.json"
        filepath = self.output_dir / filename
        
        filepath.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        
        logger.debug(f"JSON exported: {filepath}")
        
        return filepath
