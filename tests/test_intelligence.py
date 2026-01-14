"""
CUE Intelligence Tests.

Tests for the intelligence layer including STT and LLM processing.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

if TYPE_CHECKING:
    from cue.intelligence import IntentResult


# ─────────────────────────────────────────────────────────────────────────────
# DEEPGRAM SERVICE TESTS
# ─────────────────────────────────────────────────────────────────────────────


class TestDeepgramService:
    """Tests for Deepgram STT service."""

    def test_service_initialization(self) -> None:
        """Test service initializes with correct parameters."""
        with patch("deepgram.DeepgramClient"):
            from cue.intelligence import DeepgramService
            
            service = DeepgramService(
                api_key="test_key",
                model="nova-2",
                language="en-US",
            )
            
            assert service.model == "nova-2"
            assert service.language == "en-US"

    def test_empty_audio_raises_error(self) -> None:
        """Test that empty audio raises ValueError."""
        with patch("deepgram.DeepgramClient"):
            from cue.intelligence import DeepgramService
            
            service = DeepgramService(api_key="test_key")
            
            with pytest.raises(ValueError, match="empty"):
                service.transcribe(b"")


# ─────────────────────────────────────────────────────────────────────────────
# OLLAMA SERVICE TESTS
# ─────────────────────────────────────────────────────────────────────────────


class TestOllamaService:
    """Tests for Ollama LLM service."""

    def test_service_initialization(self) -> None:
        """Test service initializes with correct parameters."""
        with patch("ollama.Client"):
            from cue.intelligence import OllamaService
            
            service = OllamaService(
                model="llama3.1",
                host="http://localhost:11434",
                scratchpad_prompt="Test {transcription}",
                completion_test_prompt="Test",
                extraction_prompt="Extract",
            )
            
            assert service.model == "llama3.1"
            assert service.host == "http://localhost:11434"

    def test_scratchpad_prompt_building(self) -> None:
        """Test scratchpad prompt is correctly formatted."""
        with patch("ollama.Client"):
            from cue.intelligence import OllamaService
            
            service = OllamaService(
                scratchpad_prompt="Analyze: {transcription}",
                completion_test_prompt="",
                extraction_prompt="",
            )
            
            prompt = service._build_scratchpad_prompt("Hello world")
            
            assert "Hello world" in prompt
            assert "Analyze:" in prompt

    def test_json_extraction_from_code_block(self) -> None:
        """Test JSON extraction from markdown code blocks."""
        with patch("ollama.Client"):
            from cue.intelligence import OllamaService
            
            service = OllamaService(
                scratchpad_prompt="",
                completion_test_prompt="",
                extraction_prompt="",
            )
            
            text = '''
            Here is the result:
            ```json
            {"intent": "Schedule meeting", "action": "schedule", "entities": {}, "confidence": 0.9}
            ```
            '''
            
            result = service._extract_json(text)
            
            assert result["intent"] == "Schedule meeting"
            assert result["action"] == "schedule"
            assert result["confidence"] == 0.9

    def test_json_extraction_raw(self) -> None:
        """Test JSON extraction from raw text."""
        with patch("ollama.Client"):
            from cue.intelligence import OllamaService
            
            service = OllamaService(
                scratchpad_prompt="",
                completion_test_prompt="",
                extraction_prompt="",
            )
            
            text = 'The result is {"intent": "Test", "action": "test", "confidence": 0.8}'
            
            result = service._extract_json(text)
            
            assert result["intent"] == "Test"
            assert result["action"] == "test"

    def test_json_extraction_fallback(self) -> None:
        """Test JSON extraction fallback for invalid input."""
        with patch("ollama.Client"):
            from cue.intelligence import OllamaService
            
            service = OllamaService(
                scratchpad_prompt="",
                completion_test_prompt="",
                extraction_prompt="",
            )
            
            text = "No JSON here at all"
            
            result = service._extract_json(text)
            
            assert result["intent"] == "Unknown"
            assert result["confidence"] == 0.5

    def test_empty_transcription_returns_empty_result(self) -> None:
        """Test that empty transcription returns appropriate result."""
        with patch("ollama.Client"):
            from cue.intelligence import OllamaService
            
            service = OllamaService(
                scratchpad_prompt="",
                completion_test_prompt="",
                extraction_prompt="",
            )
            
            result = service.analyze_intent("")
            
            assert result.intent == "Empty input"
            assert result.action == "none"
            assert result.confidence == 0.0

    def test_health_check_success(self) -> None:
        """Test health check returns True when Ollama is available."""
        with patch("ollama.Client") as mock_client:
            mock_client.return_value.list.return_value = {"models": []}
            
            from cue.intelligence import OllamaService
            
            service = OllamaService(
                scratchpad_prompt="",
                completion_test_prompt="",
                extraction_prompt="",
            )
            
            assert service.health_check() is True

    def test_health_check_failure(self) -> None:
        """Test health check returns False when Ollama is unavailable."""
        with patch("ollama.Client") as mock_client:
            mock_client.return_value.list.side_effect = Exception("Connection refused")
            
            from cue.intelligence import OllamaService
            
            service = OllamaService(
                scratchpad_prompt="",
                completion_test_prompt="",
                extraction_prompt="",
            )
            
            assert service.health_check() is False


# ─────────────────────────────────────────────────────────────────────────────
# INTELLIGENCE LAYER TESTS
# ─────────────────────────────────────────────────────────────────────────────


class TestIntelligenceLayer:
    """Tests for the complete intelligence pipeline."""

    def test_layer_initialization(self) -> None:
        """Test intelligence layer initializes both services."""
        with patch("deepgram.DeepgramClient"), \
             patch("ollama.Client"):
            from cue.intelligence import IntelligenceLayer
            
            layer = IntelligenceLayer(
                deepgram_api_key="test_key",
                deepgram_config={"model": "nova-2"},
                llm_config={
                    "model": "llama3.1",
                    "scratchpad_prompt": "",
                    "completion_test_prompt": "",
                    "extraction_prompt": "",
                },
            )
            
            assert layer.stt is not None
            assert layer.llm is not None

    def test_health_check_aggregates_services(self) -> None:
        """Test health check reports status of all services."""
        with patch("deepgram.DeepgramClient"), \
             patch("ollama.Client") as mock_ollama:
            mock_ollama.return_value.list.return_value = {"models": []}
            
            from cue.intelligence import IntelligenceLayer
            
            layer = IntelligenceLayer(
                deepgram_api_key="test_key",
                deepgram_config={},
                llm_config={
                    "scratchpad_prompt": "",
                    "completion_test_prompt": "",
                    "extraction_prompt": "",
                },
            )
            
            health = layer.health_check()
            
            assert "ollama" in health
            assert "deepgram" in health
            assert health["deepgram"] is True  # Has API key


# ─────────────────────────────────────────────────────────────────────────────
# INTENT RESULT TESTS
# ─────────────────────────────────────────────────────────────────────────────


class TestIntentResult:
    """Tests for IntentResult data class."""

    def test_intent_result_creation(self, sample_intent_result: "IntentResult") -> None:
        """Test IntentResult is created correctly."""
        assert sample_intent_result.intent == "Schedule a meeting"
        assert sample_intent_result.action == "schedule"
        assert sample_intent_result.confidence == 0.92
        assert "tomorrow" in sample_intent_result.entities["date"]

    def test_intent_result_has_analysis(self, sample_intent_result: "IntentResult") -> None:
        """Test IntentResult contains analysis."""
        assert sample_intent_result.analysis is not None
        assert len(sample_intent_result.analysis) > 0
