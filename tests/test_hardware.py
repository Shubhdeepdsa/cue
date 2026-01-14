"""
CUE Hardware Tests.

Tests for hardware simulation components including HotkeyListener.
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest


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
