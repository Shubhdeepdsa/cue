"""
CUE Printer Tests.

Tests for ticket generation and saving functionality.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

if TYPE_CHECKING:
    from cue.intelligence import IntentResult


# ─────────────────────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_template() -> str:
    """Sample ticket template for testing."""
    return """
╭── 🎫 TICKET #{id} ──────────────────────────────────────────╮
│  TIMESTAMP: {timestamp}
│  TITLE: {title}
│  CATEGORY: {category}
│  NEXT ACTION: {next_action}
│  ESTIMATED TIME: {estimated_time}
│  CONFIDENCE: {confidence}%
│  
│  RAW INPUT:
│  {transcription}
╰─────────────────────────────────────────────────────────────╯
"""


@pytest.fixture
def sample_intent_result():
    """Create a sample IntentResult for testing."""
    from cue.intelligence import IntentResult
    
    return IntentResult(
        intent="Schedule a meeting",
        action="schedule",
        entities={"date": "tomorrow", "time": "10:00 AM"},
        confidence=0.92,
        raw_transcription="I want to schedule a meeting tomorrow at 10 AM",
        analysis="User wants to create a calendar event.",
        title="Schedule Meeting",
        category="Calendar",
        next_action="Create calendar event",
        estimated_time="5 minutes",
        priority="medium",
    )


@pytest.fixture
def long_transcription_intent():
    """Create an IntentResult with long transcription for wrapping tests."""
    from cue.intelligence import IntentResult
    
    long_text = (
        "This is a very long transcription that should definitely be wrapped "
        "because it exceeds the typical width of a ticket template and we need "
        "to ensure that the text wrapping functionality works correctly when "
        "processing user input that contains many words and sentences."
    )
    
    return IntentResult(
        intent="Long task",
        action="process",
        entities={},
        confidence=0.85,
        raw_transcription=long_text,
        analysis="Long analysis",
        title="Long Task Title",
        category="General",
        next_action="Process request",
        estimated_time="10 minutes",
        priority="low",
    )


# ─────────────────────────────────────────────────────────────────────────────
# TICKET PRINTER TESTS
# ─────────────────────────────────────────────────────────────────────────────


class TestTicketPrinter:
    """Tests for TicketPrinter class."""

    def test_initialization(self, tmp_path: Path, sample_template: str) -> None:
        """Test that TicketPrinter initializes correctly."""
        from cue.printer import TicketPrinter
        
        tickets_dir = tmp_path / "tickets"
        printer = TicketPrinter(tickets_dir=tickets_dir, template=sample_template)
        
        assert printer.tickets_dir == tickets_dir
        assert printer.template == sample_template
        assert tickets_dir.exists()  # Should create directory

    def test_initialization_existing_tickets(
        self, tmp_path: Path, sample_template: str
    ) -> None:
        """Test that TicketPrinter counts existing tickets."""
        from cue.printer import TicketPrinter
        
        tickets_dir = tmp_path / "tickets"
        tickets_dir.mkdir()
        
        # Create some existing ticket directories
        (tickets_dir / "ticket_20240101_000001_test").mkdir()
        (tickets_dir / "ticket_20240101_000002_test").mkdir()
        (tickets_dir / "ticket_20240101_000003_test").mkdir()
        
        printer = TicketPrinter(tickets_dir=tickets_dir, template=sample_template)
        
        assert printer._ticket_count == 3

    def test_to_snake_case(self, tmp_path: Path, sample_template: str) -> None:
        """Test snake_case conversion."""
        from cue.printer import TicketPrinter
        
        printer = TicketPrinter(tickets_dir=tmp_path, template=sample_template)
        
        assert printer._to_snake_case("Hello World") == "hello_world"
        assert printer._to_snake_case("Schedule a Meeting") == "schedule_a_meeting"

    def test_to_snake_case_special_chars(
        self, tmp_path: Path, sample_template: str
    ) -> None:
        """Test snake_case handles special characters."""
        from cue.printer import TicketPrinter
        
        printer = TicketPrinter(tickets_dir=tmp_path, template=sample_template)
        
        assert printer._to_snake_case("Hello, World!") == "hello_world"
        assert printer._to_snake_case("Test@123#Special") == "test123special"

    def test_to_snake_case_max_words(
        self, tmp_path: Path, sample_template: str
    ) -> None:
        """Test snake_case respects max_words limit."""
        from cue.printer import TicketPrinter
        
        printer = TicketPrinter(tickets_dir=tmp_path, template=sample_template)
        
        result = printer._to_snake_case("One Two Three Four Five Six", max_words=4)
        assert result == "one_two_three_four"

    def test_to_snake_case_empty(self, tmp_path: Path, sample_template: str) -> None:
        """Test snake_case handles empty string."""
        from cue.printer import TicketPrinter
        
        printer = TicketPrinter(tickets_dir=tmp_path, template=sample_template)
        
        assert printer._to_snake_case("") == "untitled"
        assert printer._to_snake_case("   ") == "untitled"

    def test_get_existing_ticket_count_empty(
        self, tmp_path: Path, sample_template: str
    ) -> None:
        """Test counting tickets in empty directory."""
        from cue.printer import TicketPrinter
        
        tickets_dir = tmp_path / "empty_tickets"
        # Don't create the directory
        printer = TicketPrinter(tickets_dir=tickets_dir, template=sample_template)
        
        # The constructor creates the directory, so count should be 0
        assert printer._ticket_count == 0

    def test_wrap_text(self, tmp_path: Path, sample_template: str) -> None:
        """Test text wrapping."""
        from cue.printer import TicketPrinter
        
        printer = TicketPrinter(tickets_dir=tmp_path, template=sample_template)
        
        long_text = "This is a very long line that should be wrapped at the specified width"
        wrapped = printer._wrap_text(long_text, width=30)
        
        # Check that lines are wrapped
        lines = wrapped.split("\n")
        assert len(lines) > 1
        for line in lines:
            assert len(line) <= 30

    def test_wrap_text_empty(self, tmp_path: Path, sample_template: str) -> None:
        """Test wrapping empty text."""
        from cue.printer import TicketPrinter
        
        printer = TicketPrinter(tickets_dir=tmp_path, template=sample_template)
        
        assert printer._wrap_text("") == ""
        assert printer._wrap_text("   ") == ""

    def test_generate_ticket_content(
        self, tmp_path: Path, sample_template: str, sample_intent_result
    ) -> None:
        """Test ticket content generation."""
        from cue.printer import TicketPrinter
        
        printer = TicketPrinter(tickets_dir=tmp_path, template=sample_template)
        
        content = printer.generate_ticket_content(
            sample_intent_result,
            ticket_id="00001",
            timestamp="2024-01-01 12:00:00 UTC",
        )
        
        assert "00001" in content
        assert "2024-01-01 12:00:00 UTC" in content
        assert "Schedule Meeting" in content
        assert "Calendar" in content
        assert "92.0%" in content

    def test_generate_ticket_content_long_transcription(
        self, tmp_path: Path, sample_template: str, long_transcription_intent
    ) -> None:
        """Test ticket content generation with long transcription."""
        from cue.printer import TicketPrinter
        
        printer = TicketPrinter(tickets_dir=tmp_path, template=sample_template)
        
        content = printer.generate_ticket_content(
            long_transcription_intent,
            ticket_id="00002",
            timestamp="2024-01-01 12:00:00 UTC",
        )
        
        # Long transcription should be wrapped with │ prefix
        assert "│  " in content

    def test_save_ticket(
        self, tmp_path: Path, sample_template: str, sample_intent_result
    ) -> None:
        """Test saving a ticket bundle."""
        from cue.printer import TicketPrinter
        
        tickets_dir = tmp_path / "tickets"
        printer = TicketPrinter(tickets_dir=tickets_dir, template=sample_template)
        
        bundle_path = printer.save_ticket(sample_intent_result)
        
        assert bundle_path.exists()
        assert bundle_path.is_dir()
        assert "ticket_" in bundle_path.name
        assert "schedule_meeting" in bundle_path.name

    def test_save_ticket_creates_txt(
        self, tmp_path: Path, sample_template: str, sample_intent_result
    ) -> None:
        """Test that save_ticket creates ticket.txt file."""
        from cue.printer import TicketPrinter
        
        printer = TicketPrinter(tickets_dir=tmp_path, template=sample_template)
        bundle_path = printer.save_ticket(sample_intent_result)
        
        txt_path = bundle_path / "ticket.txt"
        assert txt_path.exists()
        
        content = txt_path.read_text()
        assert "Schedule Meeting" in content

    def test_save_ticket_creates_json(
        self, tmp_path: Path, sample_template: str, sample_intent_result
    ) -> None:
        """Test that save_ticket creates data.json file."""
        from cue.printer import TicketPrinter
        
        printer = TicketPrinter(tickets_dir=tmp_path, template=sample_template)
        bundle_path = printer.save_ticket(sample_intent_result)
        
        json_path = bundle_path / "data.json"
        assert json_path.exists()
        
        data = json.loads(json_path.read_text())
        assert data["title"] == "Schedule Meeting"
        assert data["category"] == "Calendar"
        assert data["confidence"] == 0.92
        assert data["priority"] == "medium"

    def test_save_ticket_increments_count(
        self, tmp_path: Path, sample_template: str, sample_intent_result
    ) -> None:
        """Test that save_ticket increments ticket counter."""
        from cue.printer import TicketPrinter
        
        printer = TicketPrinter(tickets_dir=tmp_path, template=sample_template)
        
        initial_count = printer._ticket_count
        printer.save_ticket(sample_intent_result)
        assert printer._ticket_count == initial_count + 1
        
        printer.save_ticket(sample_intent_result)
        assert printer._ticket_count == initial_count + 2

    def test_print_ticket_with_save(
        self, tmp_path: Path, sample_template: str, sample_intent_result
    ) -> None:
        """Test print_ticket with save=True."""
        from cue.printer import TicketPrinter
        
        printer = TicketPrinter(tickets_dir=tmp_path, template=sample_template)
        
        result = printer.print_ticket(sample_intent_result, save=True)
        
        assert result is not None
        assert result.exists()
        assert result.is_dir()

    def test_print_ticket_preview_only(
        self, tmp_path: Path, sample_template: str, sample_intent_result
    ) -> None:
        """Test print_ticket with save=False (preview only)."""
        from cue.printer import TicketPrinter
        
        printer = TicketPrinter(tickets_dir=tmp_path, template=sample_template)
        
        with patch("cue.printer.output_console") as mock_console:
            result = printer.print_ticket(sample_intent_result, save=False)
        
        assert result is None
        # Console print should have been called for preview
        assert mock_console.print.called
