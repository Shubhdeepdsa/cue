"""
CUE Main Module.

The main entry point and event loop for the CUE system.
Implements a state machine that orchestrates:
- Hardware input (keyboard hotkey)
- Audio recording
- Speech-to-text processing
- Intent analysis
- Ticket generation
"""

from __future__ import annotations

import signal
import sys
import threading
import time
from enum import Enum, auto
from pathlib import Path
from typing import Callable

from cue.hardware import AudioRecorder, HotkeyListener, LEDSimulator, SystemState
from cue.intelligence import IntelligenceLayer, IntentResult
from cue.printer import TicketPrinter
from cue.utils import (
    CueLogger,
    console,
    format_duration,
    get_logger,
    output_console,
    print_banner,
)

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# STATE MACHINE
# ─────────────────────────────────────────────────────────────────────────────


class CueStateMachine:
    """
    Main state machine for the CUE system.
    
    Manages the lifecycle of intent capture:
    IDLE -> RECORDING -> PROCESSING -> SUCCESS/ERROR -> IDLE
    
    All state transitions are thread-safe and update the
    LED simulator display accordingly.
    """

    def __init__(
        self,
        audio_config: dict,
        led_config: dict,
        ui_config: dict,
        deepgram_api_key: str,
        deepgram_config: dict,
        llm_config: dict,
        tickets_dir: Path,
        ticket_template: str,
        hotkey: str = "space",
        debounce_ms: int = 200,
        input_mode: str = "toggle",
        double_tap_ms: int = 300,
        min_hold_duration_ms: int = 500,
    ) -> None:
        """
        Initialize the CUE state machine.
        
        Args:
            audio_config: Audio recording configuration.
            led_config: LED state configuration.
            ui_config: UI theme configuration.
            deepgram_api_key: Deepgram API key.
            deepgram_config: Deepgram STT configuration.
            llm_config: Ollama LLM configuration.
            tickets_dir: Directory for saving tickets.
            ticket_template: Template for ticket formatting.
            hotkey: Hotkey for toggle recording.
            debounce_ms: Debounce time for hotkey.
            input_mode: Input mode ("toggle" or "hold").
            double_tap_ms: Double-tap threshold for cancel.
            min_hold_duration_ms: Minimum hold duration for hold mode.
        """
        self._state = SystemState.IDLE
        self._lock = threading.Lock()
        self._running = False
        self._processing_thread: threading.Thread | None = None
        self._input_mode = input_mode
        
        # Initialize components
        self.audio = AudioRecorder(
            sample_rate=audio_config.get("sample_rate", 16000),
            channels=audio_config.get("channels", 1),
            dtype=audio_config.get("dtype", "int16"),
            max_seconds=audio_config.get("max_recording_seconds", 60),
        )
        
        self.led = LEDSimulator(led_config, ui_config)
        
        self.hotkey_listener = HotkeyListener(
            on_start=self._on_start_recording,
            on_stop=self._on_stop_recording,
            on_cancel=self._on_cancel_recording,
            hotkey=hotkey,
            mode=input_mode,
            debounce_ms=debounce_ms,
            double_tap_ms=double_tap_ms,
            min_hold_ms=min_hold_duration_ms,
        )
        
        self.intelligence = IntelligenceLayer(
            deepgram_api_key=deepgram_api_key,
            deepgram_config=deepgram_config,
            llm_config=llm_config,
        )
        
        self.printer = TicketPrinter(
            tickets_dir=tickets_dir,
            template=ticket_template,
        )
        
        logger.debug(f"CUE state machine initialized (input mode: {input_mode})")

    @property
    def state(self) -> SystemState:
        """Get current system state."""
        return self._state

    def _set_state(self, state: SystemState, message: str = "") -> None:
        """
        Thread-safe state transition.
        
        Args:
            state: New system state.
            message: Optional status message.
        """
        with self._lock:
            old_state = self._state
            self._state = state
            self.led.set_state(state, message)
            logger.info(f"State: {old_state.name} -> {state.name}")

    def _on_start_recording(self) -> None:
        """Handle start recording event from hotkey listener."""
        with self._lock:
            current_state = self._state
        
        if current_state == SystemState.IDLE:
            self._start_recording()
        elif current_state in (SystemState.ERROR, SystemState.SUCCESS):
            # Return to IDLE first, then start
            self._set_state(SystemState.IDLE)
            self._start_recording()
        else:
            logger.debug(f"Cannot start recording in state: {current_state.name}")

    def _on_stop_recording(self) -> None:
        """Handle stop recording event from hotkey listener."""
        with self._lock:
            current_state = self._state
        
        if current_state == SystemState.RECORDING:
            self._stop_recording()
        elif current_state == SystemState.PROCESSING:
            logger.warning("Cannot interrupt processing")
        else:
            logger.debug(f"Not recording, ignoring stop in state: {current_state.name}")

    def _on_cancel_recording(self) -> None:
        """Handle cancel event from hotkey listener."""
        with self._lock:
            current_state = self._state
        
        if current_state == SystemState.RECORDING:
            logger.info("Recording cancelled")
            try:
                if self.audio.is_recording:
                    self.audio.stop()  # Discard the audio
            except Exception:
                pass
            self._set_state(SystemState.IDLE, "Recording cancelled")
        else:
            logger.debug(f"Nothing to cancel in state: {current_state.name}")

    def _start_recording(self) -> None:
        """Start audio recording."""
        try:
            self._set_state(SystemState.RECORDING, "Listening...")
            self.audio.start()
            
            # Start audio level update thread
            def update_audio_level():
                while self._state == SystemState.RECORDING:
                    self.led.set_audio_level(self.audio.current_level)
                    time.sleep(0.05)
            
            threading.Thread(target=update_audio_level, daemon=True).start()
            
        except Exception as e:
            logger.error(f"Failed to start recording: {e}")
            self._set_state(SystemState.ERROR, str(e))

    def _stop_recording(self) -> None:
        """Stop recording and start processing."""
        try:
            audio_bytes = self.audio.stop()
            duration = self.audio.duration
            
            if not audio_bytes:
                logger.warning("No audio captured")
                self._set_state(SystemState.ERROR, "No audio captured")
                return
            
            # Start processing in background thread
            self._set_state(
                SystemState.PROCESSING,
                f"Processing {format_duration(duration)} of audio...",
            )
            
            self._processing_thread = threading.Thread(
                target=self._process_audio,
                args=(audio_bytes,),
                daemon=True,
            )
            self._processing_thread.start()
            
        except Exception as e:
            logger.error(f"Failed to stop recording: {e}")
            self._set_state(SystemState.ERROR, str(e))

    def _process_audio(self, audio_bytes: bytes) -> None:
        """
        Process audio through the intelligence pipeline.
        
        Args:
            audio_bytes: Recorded audio data.
        """
        try:
            # Callback to update status after transcription
            def on_transcription(result):
                self.led.set_message(f"Transcribed: {result.text[:30]}...")
            
            # Process through intelligence layer
            results = self.intelligence.process_audio(
                audio_bytes,
                on_transcription=on_transcription,
            )
            
            # Generate and print tickets
            for result in results:
                self.printer.print_ticket(result, save=True)
            
            # Success!
            count = len(results)
            msg = f"{count} Intent{'s' if count != 1 else ''} Captured"
            if count == 1:
                msg = f"Intent: {results[0].intent}"
                
            self._set_state(
                SystemState.SUCCESS,
                msg,
            )
            
            # Auto-return to IDLE after delay
            time.sleep(3)
            if self._state == SystemState.SUCCESS:
                self._set_state(SystemState.IDLE)
            
        except Exception as e:
            logger.exception(f"Processing failed: {e}")
            self._set_state(SystemState.ERROR, str(e))
            
            # Auto-return to IDLE after error display
            time.sleep(5)
            if self._state == SystemState.ERROR:
                self._set_state(SystemState.IDLE)

    def run(self) -> None:
        """
        Start the CUE system main loop.
        
        Blocks until stop() is called or SIGINT received.
        """
        logger.info("Starting CUE system...")
        
        self._running = True
        
        # Setup signal handlers
        def signal_handler(sig, frame):
            logger.info("Shutdown requested...")
            self.stop()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Print banner
        print_banner()
        output_console.print()
        
        # Health check
        health = self.intelligence.health_check()
        if not health.get("ollama"):
            logger.warning("Ollama is not available - LLM processing will fail")
            output_console.print(
                "[yellow]⚠ Warning: Ollama is not running. "
                "Start Ollama with 'ollama serve' for LLM processing.[/yellow]"
            )
            output_console.print()
        
        # Start components
        self.led.start()
        self.hotkey_listener.start()
        
        output_console.print(
            "[dim]Press SPACE to start/stop recording. Ctrl+C to exit.[/dim]"
        )
        output_console.print()
        
        # Main loop
        try:
            while self._running:
                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def stop(self) -> None:
        """Stop the CUE system gracefully."""
        if not self._running:
            return
        
        logger.info("Stopping CUE system...")
        self._running = False
        
        # Stop recording if active
        if self.audio.is_recording:
            try:
                self.audio.stop()
            except Exception:
                pass
        
        # Wait for processing to complete
        if self._processing_thread and self._processing_thread.is_alive():
            self._processing_thread.join(timeout=2.0)
        
        # Stop components
        self.hotkey_listener.stop()
        self.led.stop()
        
        output_console.print("\n[dim]CUE system stopped.[/dim]")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────


def main() -> int:
    """
    Main entry point for the CUE application.
    
    Returns:
        Exit code (0 for success, non-zero for error).
    """
    try:
        # Import settings (triggers config loading and validation)
        from cue.config import settings
        
        # Setup logging
        CueLogger.setup_logging(
            level=settings.logging.level,
            show_timestamps=settings.logging.show_timestamps,
            rich_tracebacks=settings.logging.rich_tracebacks,
            log_to_file=settings.logging.log_to_file,
            log_file_path=settings.logging.log_file_path,
        )
        
        # Ensure directories exist
        settings.ensure_directories()
        
        # Create and run state machine
        machine = CueStateMachine(
            audio_config=settings.hardware.audio.model_dump(),
            led_config={
                state.name.lower(): getattr(settings.led_states, state.name.lower()).model_dump()
                for state in SystemState
            },
            ui_config=settings.ui.model_dump(),
            deepgram_api_key=settings.deepgram_api_key,
            deepgram_config=settings.deepgram.model_dump(),
            llm_config=settings.llm.model_dump(),
            tickets_dir=settings.tickets_path,
            ticket_template=settings.paths.ticket_template,
            hotkey=settings.hardware.hotkey,
            debounce_ms=settings.hardware.debounce_ms,
            input_mode=settings.hardware.input_mode,
            double_tap_ms=settings.hardware.double_tap_ms,
            min_hold_duration_ms=settings.hardware.min_hold_duration_ms,
        )
        
        machine.run()
        
        return 0
        
    except FileNotFoundError as e:
        console.print(f"[red]Configuration Error:[/red] {e}")
        return 1
        
    except Exception as e:
        console.print(f"[red]Fatal Error:[/red] {e}")
        logger.exception("Fatal error")
        return 1


if __name__ == "__main__":
    sys.exit(main())
