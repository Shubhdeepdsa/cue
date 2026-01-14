"""
CUE Hardware Module.

Provides hardware simulation including:
- Audio recording via sounddevice
- Keyboard input via pynput
- LED state display via Rich console

All hardware interactions are threaded for non-blocking operation.
"""

from __future__ import annotations

import io
import queue
import threading
import time
from enum import Enum, auto
from typing import TYPE_CHECKING, Callable

import numpy as np
import sounddevice as sd
from pynput import keyboard
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text
from scipy.io import wavfile

from cue.utils import console, get_logger, output_console

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM STATE ENUM
# ─────────────────────────────────────────────────────────────────────────────


class SystemState(Enum):
    """Enumeration of possible system states."""

    IDLE = auto()
    RECORDING = auto()
    PROCESSING = auto()
    ERROR = auto()
    SUCCESS = auto()


# ─────────────────────────────────────────────────────────────────────────────
# AUDIO RECORDER
# ─────────────────────────────────────────────────────────────────────────────


class AudioRecorder:
    """
    Thread-safe audio recorder using sounddevice.
    
    Captures audio from the default input device in chunks and
    provides the complete recording as WAV bytes when stopped.
    
    Attributes:
        sample_rate: Audio sample rate in Hz.
        channels: Number of audio channels (1=mono, 2=stereo).
        dtype: NumPy dtype for audio samples.
        max_seconds: Maximum recording duration in seconds.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        dtype: str = "int16",
        max_seconds: int = 60,
    ) -> None:
        """
        Initialize the audio recorder.
        
        Args:
            sample_rate: Sample rate in Hz.
            channels: Number of channels.
            dtype: Audio data type.
            max_seconds: Maximum recording duration.
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.dtype = dtype
        self.max_seconds = max_seconds
        
        self._is_recording = False
        self._audio_queue: queue.Queue[NDArray[np.int16] | None] = queue.Queue()
        self._recording_thread: threading.Thread | None = None
        self._chunks: list[NDArray[np.int16]] = []
        self._lock = threading.Lock()
        self._start_time: float = 0.0
        self._current_level: float = 0.0

    @property
    def is_recording(self) -> bool:
        """Check if recording is in progress."""
        return self._is_recording

    @property
    def current_level(self) -> float:
        """Get current audio level (0.0 to 1.0)."""
        return self._current_level

    @property
    def duration(self) -> float:
        """Get current recording duration in seconds."""
        if not self._is_recording:
            return 0.0
        return time.time() - self._start_time

    def _audio_callback(
        self,
        indata: NDArray[np.int16],
        frames: int,
        time_info: dict,
        status: sd.CallbackFlags,
    ) -> None:
        """
        Callback function for sounddevice stream.
        
        Args:
            indata: Input audio data.
            frames: Number of frames.
            time_info: Timing information.
            status: Stream status flags.
        """
        if status:
            logger.warning(f"Audio callback status: {status}")
        
        # Calculate audio level for meter
        if len(indata) > 0:
            self._current_level = min(1.0, np.abs(indata).mean() / 3000)
        
        self._audio_queue.put(indata.copy())

    def _recording_worker(self) -> None:
        """Worker thread that collects audio chunks."""
        while self._is_recording:
            try:
                chunk = self._audio_queue.get(timeout=0.1)
                if chunk is not None:
                    with self._lock:
                        self._chunks.append(chunk)
                    
                    # Check max duration
                    if self.duration >= self.max_seconds:
                        logger.warning(f"Maximum recording duration ({self.max_seconds}s) reached")
                        self._is_recording = False
                        break
            except queue.Empty:
                continue

    def start(self) -> None:
        """
        Start audio recording.
        
        Raises:
            RuntimeError: If recording is already in progress.
        """
        if self._is_recording:
            raise RuntimeError("Recording is already in progress")
        
        logger.info("Starting audio recording...")
        
        self._chunks.clear()
        self._is_recording = True
        self._start_time = time.time()
        
        # Start recording worker thread
        self._recording_thread = threading.Thread(target=self._recording_worker, daemon=True)
        self._recording_thread.start()
        
        # Start audio stream
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=self.dtype,
            callback=self._audio_callback,
        )
        self._stream.start()
        
        logger.debug("Audio stream started")

    def stop(self) -> bytes:
        """
        Stop recording and return audio as WAV bytes.
        
        Returns:
            WAV file contents as bytes.
            
        Raises:
            RuntimeError: If no recording is in progress.
        """
        if not self._is_recording:
            raise RuntimeError("No recording in progress")
        
        logger.info(f"Stopping audio recording (duration: {self.duration:.1f}s)")
        
        self._is_recording = False
        self._stream.stop()
        self._stream.close()
        
        # Wait for recording thread to finish
        if self._recording_thread:
            self._recording_thread.join(timeout=1.0)
        
        # Combine all chunks
        with self._lock:
            if not self._chunks:
                logger.warning("No audio data captured")
                return b""
            
            audio_data = np.concatenate(self._chunks, axis=0)
        
        logger.debug(f"Captured {len(audio_data)} samples")
        
        # Convert to WAV bytes
        buffer = io.BytesIO()
        wavfile.write(buffer, self.sample_rate, audio_data)
        wav_bytes = buffer.getvalue()
        
        self._current_level = 0.0
        
        return wav_bytes


# ─────────────────────────────────────────────────────────────────────────────
# HOTKEY LISTENER
# ─────────────────────────────────────────────────────────────────────────────


class HotkeyListener:
    """
    Global hotkey listener using pynput.
    
    Supports two input modes:
    - toggle: Press once to start, press again to stop
    - hold: Hold key to record, release to stop
    
    Also supports double-tap to cancel recording.
    
    Attributes:
        hotkey: The key to listen for.
        mode: Input mode ("toggle" or "hold").
        double_tap_ms: Threshold for double-tap detection.
        min_hold_ms: Minimum hold duration before recording starts (hold mode).
    """

    def __init__(
        self,
        on_start: Callable[[], None],
        on_stop: Callable[[], None],
        on_cancel: Callable[[], None] | None = None,
        hotkey: str = "space",
        mode: str = "toggle",
        debounce_ms: int = 200,
        double_tap_ms: int = 300,
        min_hold_ms: int = 500,
    ) -> None:
        """
        Initialize the hotkey listener.
        
        Args:
            on_start: Function to call when recording should start.
            on_stop: Function to call when recording should stop.
            on_cancel: Function to call when recording is cancelled.
            hotkey: Key to listen for (e.g., "space", "enter").
            mode: Input mode ("toggle" or "hold").
            debounce_ms: Debounce time in milliseconds.
            double_tap_ms: Time threshold for double-tap detection.
            min_hold_ms: Minimum hold duration for hold mode.
        """
        self.on_start = on_start
        self.on_stop = on_stop
        self.on_cancel = on_cancel or (lambda: None)
        self.hotkey = hotkey
        self.mode = mode
        self.debounce_ms = debounce_ms
        self.double_tap_ms = double_tap_ms
        self.min_hold_ms = min_hold_ms
        
        self._listener: keyboard.Listener | None = None
        self._last_press_time: float = 0.0
        self._press_start_time: float = 0.0
        self._is_recording: bool = False
        self._key_held: bool = False
        self._lock = threading.Lock()
        
        self._key_map = {
            "space": keyboard.Key.space,
            "enter": keyboard.Key.enter,
            "tab": keyboard.Key.tab,
            "escape": keyboard.Key.esc,
        }

    def _on_press(self, key: keyboard.Key | keyboard.KeyCode | None) -> None:
        """
        Handle key press events.
        
        Args:
            key: The key that was pressed.
        """
        target_key = self._key_map.get(self.hotkey.lower())
        
        if key != target_key:
            return
            
        with self._lock:
            # Ignore if key is already held down (key repeat)
            if self._key_held:
                return
            
            current_time = time.time() * 1000  # Convert to ms
            time_since_last_press = current_time - self._last_press_time
            
            # Check for double-tap (cancel)
            if self._is_recording and time_since_last_press < self.double_tap_ms:
                logger.debug("Double-tap detected - cancelling")
                self._is_recording = False
                self._key_held = False
                self._last_press_time = 0  # Reset to prevent triple-tap issues
                self.on_cancel()
                return
            
            # Debounce check
            if time_since_last_press < self.debounce_ms:
                return
            
            self._last_press_time = current_time
            self._press_start_time = current_time
            self._key_held = True
            
            logger.debug(f"Hotkey '{self.hotkey}' pressed (mode: {self.mode})")
            
            if self.mode == "toggle":
                self._handle_toggle_press()
            elif self.mode == "hold":
                self._handle_hold_press()

    def _on_release(self, key: keyboard.Key | keyboard.KeyCode | None) -> None:
        """
        Handle key release events.
        
        Args:
            key: The key that was released.
        """
        target_key = self._key_map.get(self.hotkey.lower())
        
        if key != target_key:
            return
            
        with self._lock:
            if not self._key_held:
                return
                
            self._key_held = False
            hold_duration = (time.time() * 1000) - self._press_start_time
            
            logger.debug(f"Hotkey '{self.hotkey}' released (held for {hold_duration:.0f}ms)")
            
            if self.mode == "hold":
                self._handle_hold_release(hold_duration)

    def _handle_toggle_press(self) -> None:
        """Handle press event in toggle mode."""
        if not self._is_recording:
            self._is_recording = True
            self.on_start()
        else:
            self._is_recording = False
            self.on_stop()

    def _handle_hold_press(self) -> None:
        """Handle press event in hold mode."""
        if not self._is_recording:
            self._is_recording = True
            self.on_start()

    def _handle_hold_release(self, hold_duration: float) -> None:
        """
        Handle release event in hold mode.
        
        Args:
            hold_duration: How long the key was held in ms.
        """
        if self._is_recording:
            if hold_duration < self.min_hold_ms:
                # Too short - cancel instead of stop
                logger.debug(f"Hold too short ({hold_duration:.0f}ms < {self.min_hold_ms}ms) - cancelling")
                self._is_recording = False
                self.on_cancel()
            else:
                self._is_recording = False
                self.on_stop()

    def start(self) -> None:
        """Start listening for hotkey presses."""
        if self._listener is not None:
            logger.warning("Hotkey listener already running")
            return
        
        logger.info(f"Starting hotkey listener (key: {self.hotkey}, mode: {self.mode})")
        
        self._listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        )
        self._listener.start()

    def stop(self) -> None:
        """Stop the hotkey listener."""
        if self._listener is None:
            return
        
        logger.info("Stopping hotkey listener")
        self._listener.stop()
        self._listener = None


# ─────────────────────────────────────────────────────────────────────────────
# LED SIMULATOR
# ─────────────────────────────────────────────────────────────────────────────


class LEDSimulator:
    """
    Simulates LED status indicators via Rich console output.
    
    Displays a visual representation of the system state with
    colors, labels, and optional animations.
    """

    def __init__(
        self,
        led_config: dict,
        ui_config: dict,
    ) -> None:
        """
        Initialize the LED simulator.
        
        Args:
            led_config: LED state configuration from config.yaml.
            ui_config: UI theme configuration from config.yaml.
        """
        self.led_config = led_config
        self.ui_config = ui_config
        
        self._current_state = SystemState.IDLE
        self._live: Live | None = None
        self._audio_level: float = 0.0
        self._message: str = ""
        self._running = False
        self._lock = threading.Lock()
        self._update_thread: threading.Thread | None = None

    @property
    def current_state(self) -> SystemState:
        """Get the current system state."""
        return self._current_state

    def _get_state_config(self, state: SystemState) -> dict:
        """Get configuration for a specific state."""
        state_name = state.name.lower()
        return self.led_config.get(state_name, self.led_config["idle"])

    def _build_display(self) -> Panel:
        """Build the Rich display panel."""
        config = self._get_state_config(self._current_state)
        
        # Create main content table
        table = Table.grid(padding=(0, 2))
        table.add_column(justify="center", width=60)
        
        # State indicator
        state_text = Text()
        state_text.append(config["label"], style=f"bold {config['color']}")
        table.add_row(state_text)
        
        # Description
        desc_text = Text(config.get("description", ""), style="dim")
        table.add_row(desc_text)
        
        # Audio level meter (if recording)
        if self._current_state == SystemState.RECORDING and self.ui_config.get("show_audio_meter", True):
            table.add_row("")
            meter = self._build_audio_meter()
            table.add_row(meter)
        
        # Status message
        if self._message:
            table.add_row("")
            msg_text = Text(self._message, style="italic dim")
            table.add_row(msg_text)
        
        # Build panel
        panel = Panel(
            table,
            title=f"[bold]{self.ui_config.get('title', 'CUE')}[/bold]",
            border_style=config["color"],
            padding=(1, 2),
        )
        
        return panel

    def _build_audio_meter(self) -> Text:
        """Build an audio level meter visualization."""
        meter_width = 40
        filled = int(self._audio_level * meter_width)
        
        meter = Text()
        meter.append("🎤 ", style="dim")
        meter.append("█" * filled, style="bold red" if self._audio_level > 0.8 else "bold green")
        meter.append("░" * (meter_width - filled), style="dim")
        meter.append(f" {self._audio_level * 100:.0f}%", style="dim")
        
        return meter

    def _update_loop(self) -> None:
        """Background thread for updating the display."""
        refresh_rate = self.ui_config.get("refresh_rate_hz", 10)
        interval = 1.0 / refresh_rate
        
        while self._running:
            if self._live:
                with self._lock:
                    self._live.update(self._build_display())
            time.sleep(interval)

    def set_state(self, state: SystemState, message: str = "") -> None:
        """
        Set the current system state.
        
        Args:
            state: The new system state.
            message: Optional status message.
        """
        with self._lock:
            self._current_state = state
            self._message = message
        
        logger.debug(f"LED state changed to: {state.name}")

    def set_audio_level(self, level: float) -> None:
        """
        Update the audio level display.
        
        Args:
            level: Audio level from 0.0 to 1.0.
        """
        with self._lock:
            self._audio_level = max(0.0, min(1.0, level))

    def set_message(self, message: str) -> None:
        """
        Set the status message.
        
        Args:
            message: Status message to display.
        """
        with self._lock:
            self._message = message

    def start(self) -> None:
        """Start the LED simulator display."""
        logger.info("Starting LED simulator")
        
        self._running = True
        self._live = Live(
            self._build_display(),
            console=output_console,
            refresh_per_second=self.ui_config.get("refresh_rate_hz", 10),
            transient=True,  # Replace previous content instead of appending
        )
        self._live.start()
        
        # Start update thread
        self._update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self._update_thread.start()

    def stop(self) -> None:
        """Stop the LED simulator display."""
        logger.info("Stopping LED simulator")
        
        self._running = False
        
        if self._update_thread:
            self._update_thread.join(timeout=1.0)
        
        if self._live:
            self._live.stop()
            self._live = None
