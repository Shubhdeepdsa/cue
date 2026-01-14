"""
CUE Test Configuration.

Provides pytest fixtures and mocks for testing the CUE system.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Generator
from unittest.mock import MagicMock, patch

import pytest


# ─────────────────────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def temp_project_dir(tmp_path: Path) -> Generator[Path, None, None]:
    """
    Create a temporary project directory with config files.
    
    Yields:
        Path to the temporary project directory.
    """
    # Create config.yaml
    config_content = """
hardware:
  audio:
    sample_rate: 16000
    channels: 1
    dtype: "int16"
    chunk_duration_ms: 100
    max_recording_seconds: 60
  hotkey: "space"
  debounce_ms: 200

led_states:
  idle:
    color: "#00FF88"
    label: "● READY"
    description: "System ready"
  recording:
    color: "#FF3366"
    label: "◉ RECORDING"
    description: "Listening..."
    blink: true
    blink_interval_ms: 500
  processing:
    color: "#FFAA00"
    label: "◐ PROCESSING"
    description: "Analyzing..."
    animate: true
  error:
    color: "#FF0000"
    label: "✖ ERROR"
    description: "Error occurred"
  success:
    color: "#00FF00"
    label: "✔ SUCCESS"
    description: "Ticket generated"

llm:
  model: "llama3.1"
  host: "http://localhost:11434"
  timeout_seconds: 30
  temperature: 0.7
  max_tokens: 1024
  scratchpad_prompt: |
    Think about: {transcription}
  completion_test_prompt: |
    Is this complete?
  extraction_prompt: |
    Extract JSON

deepgram:
  model: "nova-2"
  language: "en-US"
  smart_format: true
  punctuate: true
  diarize: false
  utterances: false

paths:
  tickets_dir: "tickets"
  temp_audio_dir: ".temp_audio"
  ticket_template: |
    TICKET #{id}
    Intent: {intent}
    Action: {action}
    Entities: {entities}
    Confidence: {confidence}%
    Transcription: {transcription}
    Timestamp: {timestamp}

logging:
  level: "DEBUG"
  show_timestamps: false
  rich_tracebacks: false
  log_to_file: false
  log_file_path: "test.log"

ui:
  title: "CUE Test"
  border_style: "rounded"
  primary_color: "#00D4FF"
  secondary_color: "#FF6B6B"
  accent_color: "#FFE66D"
  show_audio_meter: true
  refresh_rate_hz: 10
"""
    
    config_path = tmp_path / "config.yaml"
    config_path.write_text(config_content)
    
    # Create .env file
    env_path = tmp_path / ".env"
    env_path.write_text("DEEPGRAM_API_KEY=test_api_key_12345\n")
    
    # Create tickets directory
    (tmp_path / "tickets").mkdir()
    
    # Change to temp directory
    original_cwd = os.getcwd()
    os.chdir(tmp_path)
    
    yield tmp_path
    
    # Restore original directory
    os.chdir(original_cwd)


@pytest.fixture
def mock_deepgram_client() -> Generator[MagicMock, None, None]:
    """
    Mock the Deepgram client for testing.
    
    Yields:
        Mocked DeepgramClient.
    """
    with patch("cue.intelligence.DeepgramClient") as mock:
        # Configure mock response
        mock_response = MagicMock()
        mock_response.results.channels = [
            MagicMock(
                alternatives=[
                    MagicMock(
                        transcript="Hello, I want to schedule a meeting tomorrow",
                        confidence=0.95,
                        words=[],
                    )
                ]
            )
        ]
        mock_response.results.metadata.duration = 3.5
        
        mock.return_value.listen.prerecorded.v.return_value.transcribe_file.return_value = (
            mock_response
        )
        
        yield mock


@pytest.fixture
def mock_ollama_client() -> Generator[MagicMock, None, None]:
    """
    Mock the Ollama client for testing.
    
    Yields:
        Mocked Ollama Client.
    """
    with patch("cue.intelligence.ollama.Client") as mock:
        # Configure mock responses
        mock_instance = mock.return_value
        
        # Scratchpad response
        mock_instance.generate.side_effect = [
            {"response": "The user wants to schedule a meeting tomorrow."},
            {"response": "INTENT_COMPLETE: Schedule a meeting for tomorrow"},
            {"response": '{"intent": "Schedule meeting", "action": "schedule", "entities": {"date": "tomorrow"}, "confidence": 0.9}'},
        ]
        
        mock_instance.list.return_value = {"models": [{"name": "llama3.1"}]}
        
        yield mock


@pytest.fixture
def sample_audio_bytes() -> bytes:
    """
    Generate sample audio bytes for testing.
    
    Returns:
        WAV audio data as bytes.
    """
    import io
    import numpy as np
    from scipy.io import wavfile
    
    # Generate 1 second of silence
    sample_rate = 16000
    duration = 1.0
    samples = int(sample_rate * duration)
    audio_data = np.zeros(samples, dtype=np.int16)
    
    buffer = io.BytesIO()
    wavfile.write(buffer, sample_rate, audio_data)
    
    return buffer.getvalue()


@pytest.fixture
def sample_intent_result():
    """
    Create a sample IntentResult for testing.
    
    Returns:
        IntentResult instance.
    """
    from cue.intelligence import IntentResult
    
    return IntentResult(
        intent="Schedule a meeting",
        action="schedule",
        entities={"date": "tomorrow", "time": "10:00 AM"},
        confidence=0.92,
        raw_transcription="I want to schedule a meeting tomorrow at 10 AM",
        analysis="User wants to create a calendar event.",
    )
