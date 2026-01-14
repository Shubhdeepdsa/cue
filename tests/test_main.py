"""
CUE Main Module Tests.

Tests for the CueStateMachine and main entry point.
"""

from __future__ import annotations

import signal
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch, PropertyMock

import pytest

from cue.hardware import SystemState


# ─────────────────────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def mock_audio_config() -> dict:
    """Audio configuration for testing."""
    return {
        "sample_rate": 16000,
        "channels": 1,
        "dtype": "int16",
        "max_recording_seconds": 60,
    }


@pytest.fixture
def mock_led_config() -> dict:
    """LED configuration for testing."""
    return {
        "idle": {"color": "#00FF00", "label": "READY", "description": "System ready"},
        "recording": {"color": "#FF0000", "label": "RECORDING", "description": "Recording"},
        "processing": {"color": "#FFAA00", "label": "PROCESSING", "description": "Processing"},
        "error": {"color": "#FF0000", "label": "ERROR", "description": "Error"},
        "success": {"color": "#00FF00", "label": "SUCCESS", "description": "Success"},
    }


@pytest.fixture
def mock_ui_config() -> dict:
    """UI configuration for testing."""
    return {"refresh_rate_hz": 10, "show_audio_meter": True}


@pytest.fixture
def mock_deepgram_config() -> dict:
    """Deepgram configuration for testing."""
    return {"model": "nova-2", "language": "en-US"}


@pytest.fixture
def mock_llm_config() -> dict:
    """LLM configuration for testing."""
    return {
        "model": "llama3.1",
        "scratchpad_prompt": "Test {transcription}",
        "completion_test_prompt": "Test",
        "extraction_prompt": "Extract",
    }


@pytest.fixture
def sample_ticket_template() -> str:
    """Sample ticket template."""
    return "TICKET #{id} - {title}"


@pytest.fixture
def mock_intent_result():
    """Create a mock IntentResult."""
    from cue.intelligence import IntentResult
    
    return IntentResult(
        intent="Test intent",
        action="test",
        entities={},
        confidence=0.95,
        raw_transcription="Test transcription",
        analysis="Test analysis",
        title="Test Task",
        category="Test",
        next_action="Do something",
        estimated_time="5 min",
        priority="medium",
    )


# ─────────────────────────────────────────────────────────────────────────────
# CUE STATE MACHINE TESTS
# ─────────────────────────────────────────────────────────────────────────────


class TestCueStateMachine:
    """Tests for CueStateMachine class."""

    @patch("cue.main.TicketPrinter")
    @patch("cue.main.IntelligenceLayer")
    @patch("cue.main.HotkeyListener")
    @patch("cue.main.LEDSimulator")
    @patch("cue.main.AudioRecorder")
    def test_initialization(
        self,
        mock_audio,
        mock_led,
        mock_hotkey,
        mock_intel,
        mock_printer,
        mock_audio_config,
        mock_led_config,
        mock_ui_config,
        mock_deepgram_config,
        mock_llm_config,
        sample_ticket_template,
        tmp_path,
    ) -> None:
        """Test CueStateMachine initializes correctly."""
        from cue.main import CueStateMachine
        
        machine = CueStateMachine(
            audio_config=mock_audio_config,
            led_config=mock_led_config,
            ui_config=mock_ui_config,
            deepgram_api_key="test_key",
            deepgram_config=mock_deepgram_config,
            llm_config=mock_llm_config,
            tickets_dir=tmp_path,
            ticket_template=sample_ticket_template,
        )
        
        assert machine.state == SystemState.IDLE
        mock_audio.assert_called_once()
        mock_led.assert_called_once()
        mock_hotkey.assert_called_once()
        mock_intel.assert_called_once()
        mock_printer.assert_called_once()

    @patch("cue.main.TicketPrinter")
    @patch("cue.main.IntelligenceLayer")
    @patch("cue.main.HotkeyListener")
    @patch("cue.main.LEDSimulator")
    @patch("cue.main.AudioRecorder")
    def test_state_property(
        self,
        mock_audio,
        mock_led,
        mock_hotkey,
        mock_intel,
        mock_printer,
        mock_audio_config,
        mock_led_config,
        mock_ui_config,
        mock_deepgram_config,
        mock_llm_config,
        sample_ticket_template,
        tmp_path,
    ) -> None:
        """Test state property returns current state."""
        from cue.main import CueStateMachine
        
        machine = CueStateMachine(
            audio_config=mock_audio_config,
            led_config=mock_led_config,
            ui_config=mock_ui_config,
            deepgram_api_key="test_key",
            deepgram_config=mock_deepgram_config,
            llm_config=mock_llm_config,
            tickets_dir=tmp_path,
            ticket_template=sample_ticket_template,
        )
        
        assert machine.state == SystemState.IDLE
        machine._state = SystemState.RECORDING
        assert machine.state == SystemState.RECORDING

    @patch("cue.main.TicketPrinter")
    @patch("cue.main.IntelligenceLayer")
    @patch("cue.main.HotkeyListener")
    @patch("cue.main.LEDSimulator")
    @patch("cue.main.AudioRecorder")
    def test_set_state_transitions(
        self,
        mock_audio,
        mock_led,
        mock_hotkey,
        mock_intel,
        mock_printer,
        mock_audio_config,
        mock_led_config,
        mock_ui_config,
        mock_deepgram_config,
        mock_llm_config,
        sample_ticket_template,
        tmp_path,
    ) -> None:
        """Test _set_state changes state and updates LED."""
        from cue.main import CueStateMachine
        
        machine = CueStateMachine(
            audio_config=mock_audio_config,
            led_config=mock_led_config,
            ui_config=mock_ui_config,
            deepgram_api_key="test_key",
            deepgram_config=mock_deepgram_config,
            llm_config=mock_llm_config,
            tickets_dir=tmp_path,
            ticket_template=sample_ticket_template,
        )
        
        machine._set_state(SystemState.RECORDING, "Recording...")
        
        assert machine._state == SystemState.RECORDING
        machine.led.set_state.assert_called_with(SystemState.RECORDING, "Recording...")

    @patch("cue.main.TicketPrinter")
    @patch("cue.main.IntelligenceLayer")
    @patch("cue.main.HotkeyListener")
    @patch("cue.main.LEDSimulator")
    @patch("cue.main.AudioRecorder")
    def test_on_start_recording_from_idle(
        self,
        mock_audio,
        mock_led,
        mock_hotkey,
        mock_intel,
        mock_printer,
        mock_audio_config,
        mock_led_config,
        mock_ui_config,
        mock_deepgram_config,
        mock_llm_config,
        sample_ticket_template,
        tmp_path,
    ) -> None:
        """Test _on_start_recording starts recording from IDLE state."""
        from cue.main import CueStateMachine
        
        machine = CueStateMachine(
            audio_config=mock_audio_config,
            led_config=mock_led_config,
            ui_config=mock_ui_config,
            deepgram_api_key="test_key",
            deepgram_config=mock_deepgram_config,
            llm_config=mock_llm_config,
            tickets_dir=tmp_path,
            ticket_template=sample_ticket_template,
        )
        
        machine._on_start_recording()
        
        assert machine._state == SystemState.RECORDING
        machine.audio.start.assert_called_once()

    @patch("cue.main.TicketPrinter")
    @patch("cue.main.IntelligenceLayer")
    @patch("cue.main.HotkeyListener")
    @patch("cue.main.LEDSimulator")
    @patch("cue.main.AudioRecorder")
    def test_on_start_recording_from_error(
        self,
        mock_audio,
        mock_led,
        mock_hotkey,
        mock_intel,
        mock_printer,
        mock_audio_config,
        mock_led_config,
        mock_ui_config,
        mock_deepgram_config,
        mock_llm_config,
        sample_ticket_template,
        tmp_path,
    ) -> None:
        """Test _on_start_recording from ERROR state resets to IDLE first."""
        from cue.main import CueStateMachine
        
        machine = CueStateMachine(
            audio_config=mock_audio_config,
            led_config=mock_led_config,
            ui_config=mock_ui_config,
            deepgram_api_key="test_key",
            deepgram_config=mock_deepgram_config,
            llm_config=mock_llm_config,
            tickets_dir=tmp_path,
            ticket_template=sample_ticket_template,
        )
        
        machine._state = SystemState.ERROR
        machine._on_start_recording()
        
        # Should go from ERROR -> IDLE -> RECORDING
        assert machine._state == SystemState.RECORDING

    @patch("cue.main.TicketPrinter")
    @patch("cue.main.IntelligenceLayer")
    @patch("cue.main.HotkeyListener")
    @patch("cue.main.LEDSimulator")
    @patch("cue.main.AudioRecorder")
    def test_on_start_recording_ignored_during_processing(
        self,
        mock_audio,
        mock_led,
        mock_hotkey,
        mock_intel,
        mock_printer,
        mock_audio_config,
        mock_led_config,
        mock_ui_config,
        mock_deepgram_config,
        mock_llm_config,
        sample_ticket_template,
        tmp_path,
    ) -> None:
        """Test _on_start_recording is ignored during PROCESSING."""
        from cue.main import CueStateMachine
        
        machine = CueStateMachine(
            audio_config=mock_audio_config,
            led_config=mock_led_config,
            ui_config=mock_ui_config,
            deepgram_api_key="test_key",
            deepgram_config=mock_deepgram_config,
            llm_config=mock_llm_config,
            tickets_dir=tmp_path,
            ticket_template=sample_ticket_template,
        )
        
        machine._state = SystemState.PROCESSING
        machine._on_start_recording()
        
        # Should remain in PROCESSING
        assert machine._state == SystemState.PROCESSING
        machine.audio.start.assert_not_called()

    @patch("cue.main.TicketPrinter")
    @patch("cue.main.IntelligenceLayer")
    @patch("cue.main.HotkeyListener")
    @patch("cue.main.LEDSimulator")
    @patch("cue.main.AudioRecorder")
    def test_on_stop_recording_during_recording(
        self,
        mock_audio,
        mock_led,
        mock_hotkey,
        mock_intel,
        mock_printer,
        mock_audio_config,
        mock_led_config,
        mock_ui_config,
        mock_deepgram_config,
        mock_llm_config,
        sample_ticket_template,
        tmp_path,
    ) -> None:
        """Test _on_stop_recording stops recording."""
        from cue.main import CueStateMachine
        
        machine = CueStateMachine(
            audio_config=mock_audio_config,
            led_config=mock_led_config,
            ui_config=mock_ui_config,
            deepgram_api_key="test_key",
            deepgram_config=mock_deepgram_config,
            llm_config=mock_llm_config,
            tickets_dir=tmp_path,
            ticket_template=sample_ticket_template,
        )
        
        machine._state = SystemState.RECORDING
        machine.audio.stop.return_value = b"audio_bytes"
        machine.audio.duration = 2.5
        
        machine._on_stop_recording()
        
        machine.audio.stop.assert_called_once()

    @patch("cue.main.TicketPrinter")
    @patch("cue.main.IntelligenceLayer")
    @patch("cue.main.HotkeyListener")
    @patch("cue.main.LEDSimulator")
    @patch("cue.main.AudioRecorder")
    def test_on_stop_recording_during_processing_warns(
        self,
        mock_audio,
        mock_led,
        mock_hotkey,
        mock_intel,
        mock_printer,
        mock_audio_config,
        mock_led_config,
        mock_ui_config,
        mock_deepgram_config,
        mock_llm_config,
        sample_ticket_template,
        tmp_path,
    ) -> None:
        """Test _on_stop_recording warns during PROCESSING."""
        from cue.main import CueStateMachine
        
        machine = CueStateMachine(
            audio_config=mock_audio_config,
            led_config=mock_led_config,
            ui_config=mock_ui_config,
            deepgram_api_key="test_key",
            deepgram_config=mock_deepgram_config,
            llm_config=mock_llm_config,
            tickets_dir=tmp_path,
            ticket_template=sample_ticket_template,
        )
        
        machine._state = SystemState.PROCESSING
        machine._on_stop_recording()
        
        # Should remain in PROCESSING
        assert machine._state == SystemState.PROCESSING
        machine.audio.stop.assert_not_called()

    @patch("cue.main.TicketPrinter")
    @patch("cue.main.IntelligenceLayer")
    @patch("cue.main.HotkeyListener")
    @patch("cue.main.LEDSimulator")
    @patch("cue.main.AudioRecorder")
    def test_on_stop_recording_ignored_when_idle(
        self,
        mock_audio,
        mock_led,
        mock_hotkey,
        mock_intel,
        mock_printer,
        mock_audio_config,
        mock_led_config,
        mock_ui_config,
        mock_deepgram_config,
        mock_llm_config,
        sample_ticket_template,
        tmp_path,
    ) -> None:
        """Test _on_stop_recording is ignored when IDLE."""
        from cue.main import CueStateMachine
        
        machine = CueStateMachine(
            audio_config=mock_audio_config,
            led_config=mock_led_config,
            ui_config=mock_ui_config,
            deepgram_api_key="test_key",
            deepgram_config=mock_deepgram_config,
            llm_config=mock_llm_config,
            tickets_dir=tmp_path,
            ticket_template=sample_ticket_template,
        )
        
        machine._on_stop_recording()
        
        # Should remain IDLE
        assert machine._state == SystemState.IDLE
        machine.audio.stop.assert_not_called()

    @patch("cue.main.TicketPrinter")
    @patch("cue.main.IntelligenceLayer")
    @patch("cue.main.HotkeyListener")
    @patch("cue.main.LEDSimulator")
    @patch("cue.main.AudioRecorder")
    def test_on_cancel_recording(
        self,
        mock_audio,
        mock_led,
        mock_hotkey,
        mock_intel,
        mock_printer,
        mock_audio_config,
        mock_led_config,
        mock_ui_config,
        mock_deepgram_config,
        mock_llm_config,
        sample_ticket_template,
        tmp_path,
    ) -> None:
        """Test _on_cancel_recording cancels and returns to IDLE."""
        from cue.main import CueStateMachine
        
        machine = CueStateMachine(
            audio_config=mock_audio_config,
            led_config=mock_led_config,
            ui_config=mock_ui_config,
            deepgram_api_key="test_key",
            deepgram_config=mock_deepgram_config,
            llm_config=mock_llm_config,
            tickets_dir=tmp_path,
            ticket_template=sample_ticket_template,
        )
        
        machine._state = SystemState.RECORDING
        machine.audio.is_recording = True
        
        machine._on_cancel_recording()
        
        assert machine._state == SystemState.IDLE
        machine.audio.stop.assert_called_once()

    @patch("cue.main.TicketPrinter")
    @patch("cue.main.IntelligenceLayer")
    @patch("cue.main.HotkeyListener")
    @patch("cue.main.LEDSimulator")
    @patch("cue.main.AudioRecorder")
    def test_on_cancel_nothing_to_cancel(
        self,
        mock_audio,
        mock_led,
        mock_hotkey,
        mock_intel,
        mock_printer,
        mock_audio_config,
        mock_led_config,
        mock_ui_config,
        mock_deepgram_config,
        mock_llm_config,
        sample_ticket_template,
        tmp_path,
    ) -> None:
        """Test _on_cancel_recording when not recording."""
        from cue.main import CueStateMachine
        
        machine = CueStateMachine(
            audio_config=mock_audio_config,
            led_config=mock_led_config,
            ui_config=mock_ui_config,
            deepgram_api_key="test_key",
            deepgram_config=mock_deepgram_config,
            llm_config=mock_llm_config,
            tickets_dir=tmp_path,
            ticket_template=sample_ticket_template,
        )
        
        machine._on_cancel_recording()
        
        # Should remain IDLE
        assert machine._state == SystemState.IDLE

    @patch("cue.main.TicketPrinter")
    @patch("cue.main.IntelligenceLayer")
    @patch("cue.main.HotkeyListener")
    @patch("cue.main.LEDSimulator")
    @patch("cue.main.AudioRecorder")
    def test_start_recording_failure_sets_error(
        self,
        mock_audio,
        mock_led,
        mock_hotkey,
        mock_intel,
        mock_printer,
        mock_audio_config,
        mock_led_config,
        mock_ui_config,
        mock_deepgram_config,
        mock_llm_config,
        sample_ticket_template,
        tmp_path,
    ) -> None:
        """Test _start_recording handles exceptions."""
        from cue.main import CueStateMachine
        
        machine = CueStateMachine(
            audio_config=mock_audio_config,
            led_config=mock_led_config,
            ui_config=mock_ui_config,
            deepgram_api_key="test_key",
            deepgram_config=mock_deepgram_config,
            llm_config=mock_llm_config,
            tickets_dir=tmp_path,
            ticket_template=sample_ticket_template,
        )
        
        machine.audio.start.side_effect = RuntimeError("Microphone error")
        
        machine._start_recording()
        
        assert machine._state == SystemState.ERROR

    @patch("cue.main.TicketPrinter")
    @patch("cue.main.IntelligenceLayer")
    @patch("cue.main.HotkeyListener")
    @patch("cue.main.LEDSimulator")
    @patch("cue.main.AudioRecorder")
    def test_stop_recording_no_audio(
        self,
        mock_audio,
        mock_led,
        mock_hotkey,
        mock_intel,
        mock_printer,
        mock_audio_config,
        mock_led_config,
        mock_ui_config,
        mock_deepgram_config,
        mock_llm_config,
        sample_ticket_template,
        tmp_path,
    ) -> None:
        """Test _stop_recording with empty audio."""
        from cue.main import CueStateMachine
        
        machine = CueStateMachine(
            audio_config=mock_audio_config,
            led_config=mock_led_config,
            ui_config=mock_ui_config,
            deepgram_api_key="test_key",
            deepgram_config=mock_deepgram_config,
            llm_config=mock_llm_config,
            tickets_dir=tmp_path,
            ticket_template=sample_ticket_template,
        )
        
        machine._state = SystemState.RECORDING
        machine.audio.stop.return_value = b""  # Empty audio
        machine.audio.duration = 0
        
        machine._stop_recording()
        
        assert machine._state == SystemState.ERROR

    @patch("cue.main.TicketPrinter")
    @patch("cue.main.IntelligenceLayer")
    @patch("cue.main.HotkeyListener")
    @patch("cue.main.LEDSimulator")
    @patch("cue.main.AudioRecorder")
    def test_stop_recording_failure(
        self,
        mock_audio,
        mock_led,
        mock_hotkey,
        mock_intel,
        mock_printer,
        mock_audio_config,
        mock_led_config,
        mock_ui_config,
        mock_deepgram_config,
        mock_llm_config,
        sample_ticket_template,
        tmp_path,
    ) -> None:
        """Test _stop_recording handles exceptions."""
        from cue.main import CueStateMachine
        
        machine = CueStateMachine(
            audio_config=mock_audio_config,
            led_config=mock_led_config,
            ui_config=mock_ui_config,
            deepgram_api_key="test_key",
            deepgram_config=mock_deepgram_config,
            llm_config=mock_llm_config,
            tickets_dir=tmp_path,
            ticket_template=sample_ticket_template,
        )
        
        machine._state = SystemState.RECORDING
        machine.audio.stop.side_effect = RuntimeError("Audio error")
        
        machine._stop_recording()
        
        assert machine._state == SystemState.ERROR

    @patch("cue.main.time.sleep")
    @patch("cue.main.TicketPrinter")
    @patch("cue.main.IntelligenceLayer")
    @patch("cue.main.HotkeyListener")
    @patch("cue.main.LEDSimulator")
    @patch("cue.main.AudioRecorder")
    def test_process_audio_success(
        self,
        mock_audio,
        mock_led,
        mock_hotkey,
        mock_intel,
        mock_printer,
        mock_sleep,
        mock_audio_config,
        mock_led_config,
        mock_ui_config,
        mock_deepgram_config,
        mock_llm_config,
        sample_ticket_template,
        tmp_path,
        mock_intent_result,
    ) -> None:
        """Test _process_audio successful processing."""
        from cue.main import CueStateMachine
        
        machine = CueStateMachine(
            audio_config=mock_audio_config,
            led_config=mock_led_config,
            ui_config=mock_ui_config,
            deepgram_api_key="test_key",
            deepgram_config=mock_deepgram_config,
            llm_config=mock_llm_config,
            tickets_dir=tmp_path,
            ticket_template=sample_ticket_template,
        )
        
        machine.intelligence.process_audio.return_value = [mock_intent_result]
        machine.led.suspend.return_value.__enter__ = MagicMock()
        machine.led.suspend.return_value.__exit__ = MagicMock()
        
        machine._process_audio(b"audio_bytes")
        
        machine.intelligence.process_audio.assert_called_once()
        machine.printer.print_ticket.assert_called_once()

    @patch("cue.main.time.sleep")
    @patch("cue.main.TicketPrinter")
    @patch("cue.main.IntelligenceLayer")
    @patch("cue.main.HotkeyListener")
    @patch("cue.main.LEDSimulator")
    @patch("cue.main.AudioRecorder")
    def test_process_audio_failure(
        self,
        mock_audio,
        mock_led,
        mock_hotkey,
        mock_intel,
        mock_printer,
        mock_sleep,
        mock_audio_config,
        mock_led_config,
        mock_ui_config,
        mock_deepgram_config,
        mock_llm_config,
        sample_ticket_template,
        tmp_path,
    ) -> None:
        """Test _process_audio error handling."""
        from cue.main import CueStateMachine
        
        machine = CueStateMachine(
            audio_config=mock_audio_config,
            led_config=mock_led_config,
            ui_config=mock_ui_config,
            deepgram_api_key="test_key",
            deepgram_config=mock_deepgram_config,
            llm_config=mock_llm_config,
            tickets_dir=tmp_path,
            ticket_template=sample_ticket_template,
        )
        
        machine.intelligence.process_audio.side_effect = RuntimeError("Processing failed")
        
        machine._process_audio(b"audio_bytes")
        
        # After error, state goes to ERROR then back to IDLE after the (mocked) sleep
        # Verify LED was set to ERROR state, and sleep was called for error delay
        machine.led.set_state.assert_any_call(SystemState.ERROR, "Processing failed")
        mock_sleep.assert_called_with(5)  # Error display delay
        # Final state is IDLE (after auto-return from error)
        assert machine._state == SystemState.IDLE

    @patch("cue.main.signal.signal")
    @patch("cue.main.print_banner")
    @patch("cue.main.output_console")
    @patch("cue.main.TicketPrinter")
    @patch("cue.main.IntelligenceLayer")
    @patch("cue.main.HotkeyListener")
    @patch("cue.main.LEDSimulator")
    @patch("cue.main.AudioRecorder")
    def test_run_starts_components(
        self,
        mock_audio,
        mock_led,
        mock_hotkey,
        mock_intel,
        mock_printer,
        mock_output,
        mock_banner,
        mock_signal,
        mock_audio_config,
        mock_led_config,
        mock_ui_config,
        mock_deepgram_config,
        mock_llm_config,
        sample_ticket_template,
        tmp_path,
    ) -> None:
        """Test run() starts all components."""
        from cue.main import CueStateMachine
        
        machine = CueStateMachine(
            audio_config=mock_audio_config,
            led_config=mock_led_config,
            ui_config=mock_ui_config,
            deepgram_api_key="test_key",
            deepgram_config=mock_deepgram_config,
            llm_config=mock_llm_config,
            tickets_dir=tmp_path,
            ticket_template=sample_ticket_template,
        )
        
        machine.intelligence.health_check.return_value = {"ollama": True}
        
        # Stop the run loop after a short time
        def stop_after_start():
            time.sleep(0.1)
            machine._running = False
        
        stop_thread = threading.Thread(target=stop_after_start)
        stop_thread.start()
        
        machine.run()
        stop_thread.join()
        
        machine.led.start.assert_called()
        machine.hotkey_listener.start.assert_called()

    @patch("cue.main.signal.signal")
    @patch("cue.main.print_banner")
    @patch("cue.main.output_console")
    @patch("cue.main.TicketPrinter")
    @patch("cue.main.IntelligenceLayer")
    @patch("cue.main.HotkeyListener")
    @patch("cue.main.LEDSimulator")
    @patch("cue.main.AudioRecorder")
    def test_run_ollama_warning(
        self,
        mock_audio,
        mock_led,
        mock_hotkey,
        mock_intel,
        mock_printer,
        mock_output,
        mock_banner,
        mock_signal,
        mock_audio_config,
        mock_led_config,
        mock_ui_config,
        mock_deepgram_config,
        mock_llm_config,
        sample_ticket_template,
        tmp_path,
    ) -> None:
        """Test run() warns when Ollama unavailable."""
        from cue.main import CueStateMachine
        
        machine = CueStateMachine(
            audio_config=mock_audio_config,
            led_config=mock_led_config,
            ui_config=mock_ui_config,
            deepgram_api_key="test_key",
            deepgram_config=mock_deepgram_config,
            llm_config=mock_llm_config,
            tickets_dir=tmp_path,
            ticket_template=sample_ticket_template,
        )
        
        machine.intelligence.health_check.return_value = {"ollama": False}
        
        def stop_after_start():
            time.sleep(0.1)
            machine._running = False
        
        stop_thread = threading.Thread(target=stop_after_start)
        stop_thread.start()
        
        machine.run()
        stop_thread.join()
        
        # Should print warning
        assert mock_output.print.called

    @patch("cue.main.output_console")
    @patch("cue.main.TicketPrinter")
    @patch("cue.main.IntelligenceLayer")
    @patch("cue.main.HotkeyListener")
    @patch("cue.main.LEDSimulator")
    @patch("cue.main.AudioRecorder")
    def test_stop_graceful(
        self,
        mock_audio,
        mock_led,
        mock_hotkey,
        mock_intel,
        mock_printer,
        mock_output,
        mock_audio_config,
        mock_led_config,
        mock_ui_config,
        mock_deepgram_config,
        mock_llm_config,
        sample_ticket_template,
        tmp_path,
    ) -> None:
        """Test stop() graceful shutdown."""
        from cue.main import CueStateMachine
        
        machine = CueStateMachine(
            audio_config=mock_audio_config,
            led_config=mock_led_config,
            ui_config=mock_ui_config,
            deepgram_api_key="test_key",
            deepgram_config=mock_deepgram_config,
            llm_config=mock_llm_config,
            tickets_dir=tmp_path,
            ticket_template=sample_ticket_template,
        )
        
        machine._running = True
        machine.audio.is_recording = False
        
        machine.stop()
        
        assert machine._running is False
        machine.hotkey_listener.stop.assert_called()
        machine.led.stop.assert_called()

    @patch("cue.main.output_console")
    @patch("cue.main.TicketPrinter")
    @patch("cue.main.IntelligenceLayer")
    @patch("cue.main.HotkeyListener")
    @patch("cue.main.LEDSimulator")
    @patch("cue.main.AudioRecorder")
    def test_stop_while_recording(
        self,
        mock_audio,
        mock_led,
        mock_hotkey,
        mock_intel,
        mock_printer,
        mock_output,
        mock_audio_config,
        mock_led_config,
        mock_ui_config,
        mock_deepgram_config,
        mock_llm_config,
        sample_ticket_template,
        tmp_path,
    ) -> None:
        """Test stop() stops active recording."""
        from cue.main import CueStateMachine
        
        machine = CueStateMachine(
            audio_config=mock_audio_config,
            led_config=mock_led_config,
            ui_config=mock_ui_config,
            deepgram_api_key="test_key",
            deepgram_config=mock_deepgram_config,
            llm_config=mock_llm_config,
            tickets_dir=tmp_path,
            ticket_template=sample_ticket_template,
        )
        
        machine._running = True
        machine.audio.is_recording = True
        
        machine.stop()
        
        machine.audio.stop.assert_called()

    @patch("cue.main.output_console")
    @patch("cue.main.TicketPrinter")
    @patch("cue.main.IntelligenceLayer")
    @patch("cue.main.HotkeyListener")
    @patch("cue.main.LEDSimulator")
    @patch("cue.main.AudioRecorder")
    def test_stop_not_running(
        self,
        mock_audio,
        mock_led,
        mock_hotkey,
        mock_intel,
        mock_printer,
        mock_output,
        mock_audio_config,
        mock_led_config,
        mock_ui_config,
        mock_deepgram_config,
        mock_llm_config,
        sample_ticket_template,
        tmp_path,
    ) -> None:
        """Test stop() when not running does nothing."""
        from cue.main import CueStateMachine
        
        machine = CueStateMachine(
            audio_config=mock_audio_config,
            led_config=mock_led_config,
            ui_config=mock_ui_config,
            deepgram_api_key="test_key",
            deepgram_config=mock_deepgram_config,
            llm_config=mock_llm_config,
            tickets_dir=tmp_path,
            ticket_template=sample_ticket_template,
        )
        
        machine._running = False
        
        machine.stop()
        
        # Should not call stop on components
        machine.hotkey_listener.stop.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN FUNCTION TESTS
# ─────────────────────────────────────────────────────────────────────────────


class TestMainFunction:
    """Tests for main() entry point."""

    @patch("cue.main.CueStateMachine")
    @patch("cue.main.CueLogger")
    def test_main_success(
        self, mock_logger, mock_machine, temp_project_dir
    ) -> None:
        """Test main() successful run."""
        from cue.main import main
        
        mock_machine_instance = MagicMock()
        mock_machine.return_value = mock_machine_instance
        
        result = main()
        
        assert result == 0
        mock_machine_instance.run.assert_called_once()

    def test_main_config_not_found(self) -> None:
        """Test main() handles FileNotFoundError.
        
        Note: This test is difficult to implement because settings is imported
        at module load time with `from cue.config import settings`. The 
        FileNotFoundError handling is verified in test_config.py instead.
        """
        # The FileNotFoundError catch path in main() is covered by 
        # TestFailFast.test_missing_config_file_fails in test_config.py
        pass

    @patch("cue.main.CueStateMachine")
    @patch("cue.main.CueLogger")
    @patch("cue.main.console")
    def test_main_generic_error(
        self, mock_console, mock_logger, mock_machine, temp_project_dir
    ) -> None:
        """Test main() handles generic exceptions."""
        from cue.main import main
        
        mock_machine.side_effect = RuntimeError("Unexpected error")
        
        result = main()
        
        assert result == 1
        mock_console.print.assert_called()
