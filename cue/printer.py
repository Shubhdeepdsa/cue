"""
CUE Printer Module.

Handles the generation and output of intent tickets.
Tickets are formatted using templates defined in config.yaml
and saved to the tickets/ directory.
"""

from __future__ import annotations

import json
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

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
    Supports "Ticket Bundles" (folder with .txt and .json).
    
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
        
        # Track ticket count for IDs (approximate based on folders)
        self._ticket_count = self._get_existing_ticket_count()
        
        logger.debug(f"Ticket printer initialized (dir: {tickets_dir})")

    def _to_snake_case(self, text: str, max_words: int = 4) -> str:
        """Convert text to snake_case for folder naming."""
        import re
        # Remove non-alphanumeric, lowercase, replace spaces with underscores
        clean = re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())
        words = clean.split()[:max_words]
        return '_'.join(words) if words else 'untitled'

    def _get_existing_ticket_count(self) -> int:
        """Count existing ticket bundles in the directory."""
        if not self.tickets_dir.exists():
            return 0
        # Count folders starting with ticket_
        return len(list(self.tickets_dir.glob("ticket_*")))

    def _wrap_text(self, text: str, width: int = 50) -> str:
        """Helper to wrap text for the template."""
        if not text:
            return ""
        return textwrap.fill(text, width=width)

    def _format_field(self, value: str, width: int = 51) -> str:
        """
        Format a field value to fixed width, truncating or padding as needed.
        
        Args:
            value: The field value to format.
            width: Target width for the field.
            
        Returns:
            Field value padded/truncated to exact width.
        """
        if len(value) <= width:
            return value.ljust(width)
        else:
            # Truncate with ellipsis for single-line fields
            return value[: width - 3] + "..."

    def _format_multiline_field(self, text: str, content_width: int = 51) -> str:
        """
        Format a multi-line field with proper wrapping and box borders.
        
        Args:
            text: The text to wrap and format.
            content_width: Width of content between borders.
            
        Returns:
            Formatted multi-line text compatible with template.
        """
        if not text:
            return " " * content_width
        
        # Wrap the text
        wrapped_lines = textwrap.wrap(text, width=content_width)
        
        # Pad every line to exact width
        padded_lines = [line.ljust(content_width) for line in wrapped_lines]
        
        # Join with the border sequence: 
        # End prev line with │, Newline, Start next line with │  
        # The template provides the very first "│  " and the very last "│"
        separator = f"│\n│  "
        
        return separator.join(padded_lines)

    def generate_ticket_content(
        self,
        intent_result: "IntentResult",
        ticket_id: str,
        timestamp: str,
    ) -> str:
        """
        Generate the formatted text content for a ticket.
        
        Args:
            intent_result: The processed intent data.
            ticket_id: formatted ID string.
            timestamp: formatted timestamp string.
            
        Returns:
            Formatted ticket string.
        """
        # Field widths calculated based on template structure:
        # Total box width = 57 characters inside (between ╭ and ╮)
        # Each field width = 57 - prefix_length - 1 (for closing │)
        # │  TITL:  {title}│ → prefix is 10 chars → width = 46
        # │  CREATED:   {timestamp}│ → prefix is 14 chars → width = 42
        # │  {transcription}│ → prefix is 3 chars → width = 53
        
        # Format each field to fixed width
        title_formatted = self._format_field(intent_result.title, width=46)
        category_formatted = self._format_field(intent_result.category, width=46)
        next_action_formatted = self._format_field(intent_result.next_action, width=46)
        estimated_time_formatted = self._format_field(intent_result.estimated_time, width=46)
        timestamp_formatted = self._format_field(timestamp, width=42)
        confidence_formatted = self._format_field(f"{intent_result.confidence * 100:.1f}%", width=42)
        
        # Format transcription with wrapping and borders
        transcription_formatted = self._format_multiline_field(
            intent_result.raw_transcription, content_width=53
        )

        return self.template.format(
            id=ticket_id,
            timestamp=timestamp_formatted,
            title=title_formatted,
            category=category_formatted,
            next_action=next_action_formatted,
            estimated_time=estimated_time_formatted,
            confidence=confidence_formatted,
            transcription=transcription_formatted,
        )

    def save_ticket(
        self,
        intent_result: "IntentResult",
    ) -> Path:
        """
        Generate and save a ticket bundle (Folder with Text + JSON).
        
        Args:
            intent_result: The processed intent data.
            
        Returns:
            Path to the ticket DIRECTORY.
        """
        self._ticket_count += 1
        ticket_id = f"{self._ticket_count:05d}"
        timestamp_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        file_timestamp = get_timestamp_id()
        
        # 1. Create Bundle Directory with descriptive name
        title_slug = self._to_snake_case(intent_result.title)
        bundle_dir_name = f"ticket_{file_timestamp}_{ticket_id}_{title_slug}"
        bundle_dir = self.tickets_dir / bundle_dir_name
        bundle_dir.mkdir(parents=True, exist_ok=True)
        
        # 2. Generate and Save Text Ticket
        ticket_content = self.generate_ticket_content(intent_result, ticket_id, timestamp_str)
        txt_path = bundle_dir / "ticket.txt"
        txt_path.write_text(ticket_content, encoding="utf-8")
        
        # 3. Generate and Save JSON Data
        json_data = {
            "id": ticket_id,
            "timestamp": timestamp_str,
            "title": intent_result.title,
            "category": intent_result.category,
            "next_action": intent_result.next_action,
            "estimated_time": intent_result.estimated_time,
            "priority": intent_result.priority,
            "confidence": intent_result.confidence,
            "raw_transcription": intent_result.raw_transcription,
            "analysis": intent_result.analysis,
        }
        json_path = bundle_dir / "data.json"
        json_path.write_text(json.dumps(json_data, indent=2, ensure_ascii=False), encoding="utf-8")
        
        logger.info(f"Ticket bundle saved: {bundle_dir}")
        
        return bundle_dir

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
            Path to saved bundle if save=True, else None.
        """
        # For console print, we generate a transient content string
        # We reuse the ID tracking logic only if saving? 
        # No, printing usually implies we just generated it.
        # But if we print WITHOUT saving, we shouldn't increment ID in a real system,
        # but here we usually do both.
        # For safety, we only increment ID if we save. 
        # If we just print, we might use a placeholder or peek.
        # BUT existing logic calls print_ticket(save=True).
        
        if save:
            return self.save_ticket(intent_result)
        else:
            # Just print to console (preview)
            # We construct a fake ID for preview
            preview_content = self.generate_ticket_content(
                intent_result, "PREVIEW", datetime.now().strftime("%H:%M:%S")
            )
            output_console.print()
            output_console.print(
                Panel(
                    Text(preview_content, style="bold"),
                    title="[bold green]🎫 TICKET PREVIEW[/bold green]",
                    border_style="green",
                    padding=(1, 2),
                )
            )
            return None
