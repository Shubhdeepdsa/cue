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
        # Format fields with wrapping
        # We assume the template handles indentation if needed, or we just wrap.
        # For the specific box template, simple wrapping might break vertical lines 
        # unless the template is designed for it. 
        # The user's template has box drawing chars. 
        # To strictly maintain the box, we would need sophisticated block formatting.
        # For now, we will use simple placement or just fill.
        # The user asked for "word wrap basically if the raw input is long".
        # The provided template put {transcription} inside the box.
        # If we just replace it, newlines will break the box sides.
        # We will wrap the text, but to keep the box perfect is hard without a layout engine.
        # However, for an "80mm paper" sim, maybe we just wrap and don't worry about closing the right side 
        # perfectly on every line, OR we assume the template is flexible.
        # Let's try to fit it into the visual block.
        
        # NOTE: The template in config.yaml is:
        # ╭── 🎫 CUE TICKET #{id} ...
        # ...
        # │  RAW INPUT:
        # │  {transcription}
        # ...
        # we will wrap transcription to fit width.
        
        wrapped_transcription = self._wrap_text(intent_result.raw_transcription, width=54)
        # Indent subsequent lines of transcription to align with first line if needed, 
        # but the template puts it on a new line.
        # We add a left margin to wrapped lines to look good.
        wrapped_transcription = wrapped_transcription.replace("\n", "\n│  ")

        return self.template.format(
            id=ticket_id,
            timestamp=timestamp,
            title=intent_result.title,
            category=intent_result.category,
            next_action=intent_result.next_action,
            estimated_time=intent_result.estimated_time,
            confidence=f"{intent_result.confidence * 100:.1f}",
            transcription=wrapped_transcription,
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
