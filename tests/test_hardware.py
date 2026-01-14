"""
CUE Hardware Tests.

Tests for hardware simulation components including HotkeyListener.
"""

from __future__ import annotations

import io
import queue
import threading
import time
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pytest


# ─────────────────────────────────────────────────────────────────────────────
# AUDIO RECORDER TESTS
# ─────────────────────────────────────────────────────────────────────────────


class TestAudioRecorder:
    """Tests for AudioRecorder class."""

    @patch("sounddevice.InputStream")
    def test_initialization(self, mock_stream) -> None:
        """Test AudioRecorder initializes with defaults."""
        from cue.hardware import AudioRecorder
        
        recorder = AudioRecorder()
        
        assert recorder.sample_rate == 16000
        assert recorder.channels == 1
        assert recorder.dtype == "int16"
        assert recorder.max_seconds == 60

    @patch("sounddevice.InputStream")
    def test_initialization_custom_params(self, mock_stream) -> None:
        """Test AudioRecorder with custom parameters."""
        from cue.hardware import AudioRecorder
        
        recorder = AudioRecorder(
            sample_rate=44100,
            channels=2,
            dtype="float32",
            max_seconds=30,
        )
        
        assert recorder.sample_rate == 44100
        assert recorder.channels == 2
        assert recorder.dtype == "float32"
        assert recorder.max_seconds == 30

    @patch("sounddevice.InputStream")
    def test_is_recording_property(self, mock_stream) -> None:
        """Test is_recording property."""
        from cue.hardware import AudioRecorder
        
        recorder = AudioRecorder()
        
        assert recorder.is_recording is False
        
        recorder._is_recording = True
        assert recorder.is_recording is True

    @patch("sounddevice.InputStream")
    def test_current_level_property(self, mock_stream) -> None:
        """Test current_level property."""
        from cue.hardware import AudioRecorder
        
        recorder = AudioRecorder()
        
        assert recorder.current_level == 0.0
        
        recorder._current_level = 0.75
        assert recorder.current_level == 0.75

    @patch("sounddevice.InputStream")
    def test_duration_not_recording(self, mock_stream) -> None:
        """Test duration returns 0 when not recording."""
        from cue.hardware import AudioRecorder
        
        recorder = AudioRecorder()
        
        assert recorder.duration == 0.0

    @patch("sounddevice.InputStream")
    def test_duration_while_recording(self, mock_stream) -> None:
        """Test duration returns elapsed time while recording."""
        from cue.hardware import AudioRecorder
        
        recorder = AudioRecorder()
        recorder._is_recording = True
        recorder._start_time = time.time() - 5.0
        
        # Should be approximately 5 seconds
        assert 4.9 <= recorder.duration <= 5.1

    @patch("sounddevice.InputStream")
    def test_start_recording(self, mock_stream) -> None:
        """Test start() begins recording."""
        from cue.hardware import AudioRecorder
        
        recorder = AudioRecorder()
        
        recorder.start()
        
        assert recorder._is_recording is True
        mock_stream.assert_called_once()
        mock_stream.return_value.start.assert_called_once()

    @patch("sounddevice.InputStream")
    def test_start_already_recording(self, mock_stream) -> None:
        """Test start() raises error if already recording."""
        from cue.hardware import AudioRecorder
        
        recorder = AudioRecorder()
        recorder._is_recording = True
        
        with pytest.raises(RuntimeError, match="already in progress"):
            recorder.start()

    @patch("sounddevice.InputStream")
    def test_stop_not_recording(self, mock_stream) -> None:
        """Test stop() raises error if not recording."""
        from cue.hardware import AudioRecorder
        
        recorder = AudioRecorder()
        
        with pytest.raises(RuntimeError, match="No recording"):
            recorder.stop()

    @patch("sounddevice.InputStream")
    def test_stop_recording_returns_wav(self, mock_stream) -> None:
        """Test stop() returns WAV bytes."""
        from cue.hardware import AudioRecorder
        
        recorder = AudioRecorder()
        recorder._is_recording = True
        recorder._start_time = time.time()
        recorder._stream = mock_stream.return_value
        
        # Add some audio data
        audio_chunk = np.zeros((1600,), dtype=np.int16)
        recorder._chunks.append(audio_chunk)
        
        result = recorder.stop()
        
        assert recorder._is_recording is False
        assert isinstance(result, bytes)
        assert len(result) > 0  # Should have WAV header + data

    @patch("sounddevice.InputStream")
    def test_stop_recording_no_chunks(self, mock_stream) -> None:
        """Test stop() with no audio chunks returns empty bytes."""
        from cue.hardware import AudioRecorder
        
        recorder = AudioRecorder()
        recorder._is_recording = True
        recorder._start_time = time.time()
        recorder._stream = mock_stream.return_value
        recorder._chunks = []
        
        result = recorder.stop()
        
        assert result == b""

    @patch("sounddevice.InputStream")
    def test_audio_callback_updates_level(self, mock_stream) -> None:
        """Test _audio_callback updates current level."""
        from cue.hardware import AudioRecorder
        import sounddevice as sd
        
        recorder = AudioRecorder()
        
        # Create test audio data
        audio_data = np.array([[1000], [2000], [3000]], dtype=np.int16)
        
        recorder._audio_callback(audio_data, 3, {}, sd.CallbackFlags())
        
        # Level should be updated
        assert recorder._current_level > 0

    @patch("sounddevice.InputStream")
    def test_audio_callback_logs_status(self, mock_stream) -> None:
        """Test _audio_callback logs warning on status."""
        from cue.hardware import AudioRecorder
        import sounddevice as sd
        
        recorder = AudioRecorder()
        
        # Create test audio data
        audio_data = np.array([[1000]], dtype=np.int16)
        
        # Create a mock status that evaluates to True
        status = MagicMock()
        status.__bool__ = lambda s: True
        
        # This should log a warning but not raise
        recorder._audio_callback(audio_data, 1, {}, status)


# ─────────────────────────────────────────────────────────────────────────────
# HOTKEY LISTENER TESTS
# ─────────────────────────────────────────────────────────────────────────────


class TestHotkeyListener:
    """Tests for HotkeyListener with toggle and hold modes."""

    def test_toggle_mode_start_stop(self) -> None:
        """Test toggle mode: first press starts, second press stops."""
        with patch("pynput.keyboard.Listener"):
            from cue.hardware import HotkeyListener
            
            start_called = []
            stop_called = []
            
            listener = HotkeyListener(
                on_start=lambda: start_called.append(True),
                on_stop=lambda: stop_called.append(True),
                on_cancel=lambda: None,
                mode="toggle",
                debounce_ms=50,
            )
            
            # Simulate first press
            listener._key_held = False
            listener._is_recording = False
            listener._last_press_time = 0
            listener._handle_toggle_press()
            
            assert len(start_called) == 1
            assert len(stop_called) == 0
            assert listener._is_recording is True
            
            # Simulate second press
            listener._handle_toggle_press()
            
            assert len(start_called) == 1
            assert len(stop_called) == 1
            assert listener._is_recording is False

    def test_hold_mode_start_stop(self) -> None:
        """Test hold mode: press starts, release stops."""
        with patch("pynput.keyboard.Listener"):
            from cue.hardware import HotkeyListener
            
            start_called = []
            stop_called = []
            
            listener = HotkeyListener(
                on_start=lambda: start_called.append(True),
                on_stop=lambda: stop_called.append(True),
                on_cancel=lambda: None,
                mode="hold",
                min_hold_ms=100,
            )
            
            # Simulate press
            listener._is_recording = False
            listener._handle_hold_press()
            
            assert len(start_called) == 1
            assert listener._is_recording is True
            
            # Simulate release after sufficient hold time
            listener._handle_hold_release(hold_duration=200)
            
            assert len(stop_called) == 1
            assert listener._is_recording is False

    def test_hold_mode_short_press_cancels(self) -> None:
        """Test hold mode: short press cancels instead of stopping."""
        with patch("pynput.keyboard.Listener"):
            from cue.hardware import HotkeyListener
            
            cancel_called = []
            stop_called = []
            
            listener = HotkeyListener(
                on_start=lambda: None,
                on_stop=lambda: stop_called.append(True),
                on_cancel=lambda: cancel_called.append(True),
                mode="hold",
                min_hold_ms=500,
            )
            
            # Start recording
            listener._is_recording = True
            
            # Release after short hold (less than min_hold_ms)
            listener._handle_hold_release(hold_duration=100)
            
            assert len(cancel_called) == 1
            assert len(stop_called) == 0
            assert listener._is_recording is False

    def test_initialization_with_defaults(self) -> None:
        """Test that HotkeyListener initializes with correct defaults."""
        with patch("pynput.keyboard.Listener"):
            from cue.hardware import HotkeyListener
            
            listener = HotkeyListener(
                on_start=lambda: None,
                on_stop=lambda: None,
            )
            
            assert listener.mode == "toggle"
            assert listener.hotkey == "space"
            assert listener.debounce_ms == 200
            assert listener.double_tap_ms == 300
            assert listener.min_hold_ms == 500

    def test_key_repeat_ignored(self) -> None:
        """Test that key repeat (holding key) doesn't trigger multiple starts."""
        with patch("pynput.keyboard.Listener"):
            from cue.hardware import HotkeyListener
            from pynput import keyboard
            
            start_count = []
            
            listener = HotkeyListener(
                on_start=lambda: start_count.append(True),
                on_stop=lambda: None,
                mode="toggle",
                debounce_ms=50,
            )
            
            # First press
            listener._last_press_time = 0
            listener._on_press(keyboard.Key.space)
            
            assert len(start_count) == 1
            
            # Simulated key repeat (key still held)
            listener._on_press(keyboard.Key.space)
            
            # Should still be 1 since key is held
            assert len(start_count) == 1

    def test_double_tap_cancels(self) -> None:
        """Test double-tap cancels recording."""
        with patch("pynput.keyboard.Listener"):
            from cue.hardware import HotkeyListener
            from pynput import keyboard
            
            cancel_called = []
            
            listener = HotkeyListener(
                on_start=lambda: None,
                on_stop=lambda: None,
                on_cancel=lambda: cancel_called.append(True),
                mode="toggle",
                debounce_ms=50,
                double_tap_ms=300,
            )
            
            # First press to start recording
            listener._last_press_time = 0
            listener._key_held = False
            listener._on_press(keyboard.Key.space)
            
            # Release
            listener._on_release(keyboard.Key.space)
            
            # Second quick press (within double_tap_ms)
            listener._on_press(keyboard.Key.space)
            
            assert len(cancel_called) == 1

    def test_wrong_key_ignored(self) -> None:
        """Test that wrong keys are ignored."""
        with patch("pynput.keyboard.Listener"):
            from cue.hardware import HotkeyListener
            from pynput import keyboard
            
            start_count = []
            
            listener = HotkeyListener(
                on_start=lambda: start_count.append(True),
                on_stop=lambda: None,
                hotkey="space",
            )
            
            listener._on_press(keyboard.Key.enter)
            
            assert len(start_count) == 0

    def test_start_listener(self) -> None:
        """Test start() creates and starts listener."""
        with patch("pynput.keyboard.Listener") as mock_listener:
            from cue.hardware import HotkeyListener
            
            listener = HotkeyListener(
                on_start=lambda: None,
                on_stop=lambda: None,
            )
            
            listener.start()
            
            mock_listener.assert_called_once()
            mock_listener.return_value.start.assert_called_once()

    def test_start_already_running(self) -> None:
        """Test start() warns if already running."""
        with patch("pynput.keyboard.Listener") as mock_listener:
            from cue.hardware import HotkeyListener
            
            listener = HotkeyListener(
                on_start=lambda: None,
                on_stop=lambda: None,
            )
            
            listener._listener = MagicMock()
            
            # Should not raise, just warn
            listener.start()
            
            # Original listener should not be replaced
            mock_listener.assert_not_called()

    def test_stop_listener(self) -> None:
        """Test stop() stops listener."""
        with patch("pynput.keyboard.Listener"):
            from cue.hardware import HotkeyListener
            
            listener = HotkeyListener(
                on_start=lambda: None,
                on_stop=lambda: None,
            )
            
            mock_listener = MagicMock()
            listener._listener = mock_listener
            
            listener.stop()
            
            mock_listener.stop.assert_called_once()
            assert listener._listener is None

    def test_stop_when_not_running(self) -> None:
        """Test stop() does nothing when not running."""
        with patch("pynput.keyboard.Listener"):
            from cue.hardware import HotkeyListener
            
            listener = HotkeyListener(
                on_start=lambda: None,
                on_stop=lambda: None,
            )
            
            # Should not raise
            listener.stop()

    def test_on_release_wrong_key(self) -> None:
        """Test _on_release ignores wrong keys."""
        with patch("pynput.keyboard.Listener"):
            from cue.hardware import HotkeyListener
            from pynput import keyboard
            
            listener = HotkeyListener(
                on_start=lambda: None,
                on_stop=lambda: None,
                mode="hold",
            )
            
            listener._key_held = True
            listener._on_release(keyboard.Key.enter)
            
            # key_held should remain unchanged
            assert listener._key_held is True


# ─────────────────────────────────────────────────────────────────────────────
# LED SIMULATOR TESTS
# ─────────────────────────────────────────────────────────────────────────────


class TestLEDSimulator:
    """Tests for LEDSimulator display."""

    def test_state_transitions(self) -> None:
        """Test that state transitions update correctly."""
        with patch("cue.hardware.Live"):
            from cue.hardware import LEDSimulator, SystemState
            
            led_config = {
                "idle": {"color": "#00FF00", "label": "READY"},
                "recording": {"color": "#FF0000", "label": "RECORDING"},
            }
            ui_config = {"refresh_rate_hz": 10}
            
            simulator = LEDSimulator(led_config, ui_config)
            
            assert simulator.current_state == SystemState.IDLE
            
            simulator.set_state(SystemState.RECORDING, "Listening...")
            
            assert simulator.current_state == SystemState.RECORDING

    def test_audio_level_bounds(self) -> None:
        """Test that audio level is clamped between 0 and 1."""
        with patch("cue.hardware.Live"):
            from cue.hardware import LEDSimulator
            
            led_config = {"idle": {"color": "#00FF00", "label": "READY"}}
            ui_config = {}
            
            simulator = LEDSimulator(led_config, ui_config)
            
            simulator.set_audio_level(1.5)
            assert simulator._audio_level == 1.0
            
            simulator.set_audio_level(-0.5)
            assert simulator._audio_level == 0.0
            
            simulator.set_audio_level(0.5)
            assert simulator._audio_level == 0.5

    def test_set_message(self) -> None:
        """Test set_message updates message."""
        with patch("cue.hardware.Live"):
            from cue.hardware import LEDSimulator
            
            led_config = {"idle": {"color": "#00FF00", "label": "READY"}}
            ui_config = {}
            
            simulator = LEDSimulator(led_config, ui_config)
            
            simulator.set_message("Test message")
            
            assert simulator._message == "Test message"

    def test_get_state_config(self) -> None:
        """Test _get_state_config returns correct config."""
        with patch("cue.hardware.Live"):
            from cue.hardware import LEDSimulator, SystemState
            
            led_config = {
                "idle": {"color": "#00FF00", "label": "READY"},
                "recording": {"color": "#FF0000", "label": "RECORDING"},
            }
            ui_config = {}
            
            simulator = LEDSimulator(led_config, ui_config)
            
            config = simulator._get_state_config(SystemState.RECORDING)
            
            assert config["color"] == "#FF0000"
            assert config["label"] == "RECORDING"

    def test_build_display(self) -> None:
        """Test _build_display returns a Table."""
        with patch("cue.hardware.Live"):
            from cue.hardware import LEDSimulator
            from rich.table import Table
            
            led_config = {
                "idle": {"color": "#00FF00", "label": "READY", "description": "Ready"},
            }
            ui_config = {"show_audio_meter": True}
            
            simulator = LEDSimulator(led_config, ui_config)
            
            display = simulator._build_display()
            
            assert isinstance(display, Table)

    def test_build_audio_meter(self) -> None:
        """Test _build_audio_meter returns Text."""
        with patch("cue.hardware.Live"):
            from cue.hardware import LEDSimulator
            from rich.text import Text
            
            led_config = {"idle": {"color": "#00FF00", "label": "READY"}}
            ui_config = {}
            
            simulator = LEDSimulator(led_config, ui_config)
            simulator._audio_level = 0.5
            
            meter = simulator._build_audio_meter()
            
            assert isinstance(meter, Text)

    def test_start_simulator(self) -> None:
        """Test start() begins display."""
        with patch("cue.hardware.Live") as mock_live:
            from cue.hardware import LEDSimulator
            
            led_config = {"idle": {"color": "#00FF00", "label": "READY"}}
            ui_config = {}
            
            simulator = LEDSimulator(led_config, ui_config)
            
            simulator.start()
            
            assert simulator._running is True
            mock_live.assert_called()

    def test_start_already_running(self) -> None:
        """Test start() does nothing if already running."""
        with patch("cue.hardware.Live") as mock_live:
            from cue.hardware import LEDSimulator
            
            led_config = {"idle": {"color": "#00FF00", "label": "READY"}}
            ui_config = {}
            
            simulator = LEDSimulator(led_config, ui_config)
            simulator._running = True
            
            simulator.start()
            
            # Live should not be called again
            mock_live.assert_not_called()

    def test_stop_simulator(self) -> None:
        """Test stop() ends display."""
        with patch("cue.hardware.Live"):
            from cue.hardware import LEDSimulator
            
            led_config = {"idle": {"color": "#00FF00", "label": "READY"}}
            ui_config = {}
            
            simulator = LEDSimulator(led_config, ui_config)
            simulator._running = True
            mock_live = MagicMock()
            simulator._live = mock_live
            
            simulator.stop()
            
            assert simulator._running is False
            mock_live.stop.assert_called_once()

    def test_suspend_context_manager(self) -> None:
        """Test suspend() context manager."""
        with patch("cue.hardware.Live"):
            from cue.hardware import LEDSimulator
            
            led_config = {"idle": {"color": "#00FF00", "label": "READY"}}
            ui_config = {}
            
            simulator = LEDSimulator(led_config, ui_config)
            simulator._running = True
            simulator._live = MagicMock()
            
            with simulator.suspend():
                simulator._live.stop.assert_called_once()
            
            simulator._live.start.assert_called_once()

    def test_suspend_not_running(self) -> None:
        """Test suspend() when not running."""
        with patch("cue.hardware.Live"):
            from cue.hardware import LEDSimulator
            
            led_config = {"idle": {"color": "#00FF00", "label": "READY"}}
            ui_config = {}
            
            simulator = LEDSimulator(led_config, ui_config)
            simulator._running = False
            
            # Should not raise
            with simulator.suspend():
                pass


# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM STATE TESTS
# ─────────────────────────────────────────────────────────────────────────────


class TestSystemState:
    """Tests for SystemState enum."""

    def test_all_states_exist(self) -> None:
        """Test all expected states exist."""
        from cue.hardware import SystemState
        
        assert hasattr(SystemState, "IDLE")
        assert hasattr(SystemState, "RECORDING")
        assert hasattr(SystemState, "PROCESSING")
        assert hasattr(SystemState, "ERROR")
        assert hasattr(SystemState, "SUCCESS")

