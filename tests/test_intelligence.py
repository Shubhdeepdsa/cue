"""
CUE Intelligence Tests.

Tests for the intelligence layer including STT and LLM processing.
"""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

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
        from cue.intelligence import DeepgramService
        
        service = DeepgramService(api_key="test_key")
        
        with pytest.raises(ValueError, match="empty"):
            service.transcribe(b"")

    @pytest.mark.asyncio
    async def test_transcribe_async_success(self) -> None:
        """Test successful async transcription."""
        from cue.intelligence import DeepgramService
        
        service = DeepgramService(api_key="test_key")
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": {
                "channels": [
                    {
                        "alternatives": [
                            {"transcript": "Hello world", "confidence": 0.95}
                        ]
                    }
                ]
            },
            "metadata": {"duration": 1.5},
        }
        
        with patch("httpx.AsyncClient") as mock_client:
            mock_context = AsyncMock()
            mock_context.__aenter__.return_value.post = AsyncMock(return_value=mock_response)
            mock_client.return_value = mock_context
            
            result = await service.transcribe_async(b"audio_data")
            
            assert result.text == "Hello world"
            assert result.confidence == 0.95
            assert result.duration_seconds == 1.5

    @pytest.mark.asyncio
    async def test_transcribe_async_api_error(self) -> None:
        """Test async transcription handles API errors."""
        from cue.intelligence import DeepgramService
        
        service = DeepgramService(api_key="test_key")
        
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = "Bad request"
        
        with patch("httpx.AsyncClient") as mock_client:
            mock_context = AsyncMock()
            mock_context.__aenter__.return_value.post = AsyncMock(return_value=mock_response)
            mock_client.return_value = mock_context
            
            with pytest.raises(ValueError, match="Deepgram API error"):
                await service.transcribe_async(b"audio_data")

    @pytest.mark.asyncio
    async def test_transcribe_async_no_channels(self) -> None:
        """Test async transcription handles missing channels."""
        from cue.intelligence import DeepgramService
        
        service = DeepgramService(api_key="test_key")
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"results": {"channels": []}}
        
        with patch("httpx.AsyncClient") as mock_client:
            mock_context = AsyncMock()
            mock_context.__aenter__.return_value.post = AsyncMock(return_value=mock_response)
            mock_client.return_value = mock_context
            
            with pytest.raises(ValueError, match="No transcription channels"):
                await service.transcribe_async(b"audio_data")

    @pytest.mark.asyncio
    async def test_transcribe_async_no_alternatives(self) -> None:
        """Test async transcription handles missing alternatives."""
        from cue.intelligence import DeepgramService
        
        service = DeepgramService(api_key="test_key")
        
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": {
                "channels": [{"alternatives": []}]
            }
        }
        
        with patch("httpx.AsyncClient") as mock_client:
            mock_context = AsyncMock()
            mock_context.__aenter__.return_value.post = AsyncMock(return_value=mock_response)
            mock_client.return_value = mock_context
            
            with pytest.raises(ValueError, match="No transcription alternatives"):
                await service.transcribe_async(b"audio_data")

    def test_transcribe_sync(self) -> None:
        """Test synchronous transcribe wrapper."""
        from cue.intelligence import DeepgramService
        
        service = DeepgramService(api_key="test_key")
        
        with patch.object(service, "transcribe_async") as mock_async:
            mock_result = MagicMock()
            mock_async.return_value = mock_result
            
            with patch("asyncio.run") as mock_run:
                mock_run.return_value = mock_result
                
                result = service.transcribe(b"audio_data")
                
                mock_run.assert_called_once()


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

    def test_json_extraction_array(self) -> None:
        """Test JSON extraction of array."""
        with patch("ollama.Client"):
            from cue.intelligence import OllamaService
            
            service = OllamaService(
                scratchpad_prompt="",
                completion_test_prompt="",
                extraction_prompt="",
            )
            
            text = '[{"intent": "Task 1"}, {"intent": "Task 2"}]'
            
            result = service._extract_json(text)
            
            assert isinstance(result, list)
            assert len(result) == 2

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
            
            assert result[0]["intent"] == "Unknown"
            assert result[0]["confidence"] == 0.5

    def test_json_extraction_tickets_wrapper(self) -> None:
        """Test JSON extraction with 'tickets' key wrapper (multi-ticket support)."""
        with patch("ollama.Client"):
            from cue.intelligence import OllamaService
            
            service = OllamaService(
                scratchpad_prompt="",
                completion_test_prompt="",
                extraction_prompt="",
            )
            
            # Simulate LLM response with "tickets" wrapper
            text = '''{"tickets": [
                {"title": "Pick up mom", "category": "Errand"},
                {"title": "Walk the dog", "category": "Personal"}
            ]}'''
            
            result = service._extract_json(text)
            
            # The result should be the dict with tickets key
            assert "tickets" in result
            assert len(result["tickets"]) == 2
            assert result["tickets"][0]["title"] == "Pick up mom"
            assert result["tickets"][1]["title"] == "Walk the dog"

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
            
            assert result[0].intent == "Empty input"
            assert result[0].action == "none"
            assert result[0].confidence == 0.0

    @pytest.mark.asyncio
    async def test_analyze_intent_async_success(self) -> None:
        """Test successful async intent analysis."""
        with patch("ollama.Client") as mock_client:
            from cue.intelligence import OllamaService
            
            # Mock generate responses
            mock_client.return_value.generate.return_value = {
                "response": '{"intent": "Test", "action": "test", "confidence": 0.9}'
            }
            
            service = OllamaService(
                scratchpad_prompt="{transcription}",
                completion_test_prompt="Complete",
                extraction_prompt="Extract",
            )
            
            # Patch asyncio.to_thread to run synchronously
            async def mock_to_thread(func, *args, **kwargs):
                return func(*args, **kwargs)
            
            with patch("asyncio.to_thread", side_effect=mock_to_thread):
                result = await service.analyze_intent_async("Test input")
                
                assert len(result) >= 1
                assert result[0].raw_transcription == "Test input"

    @pytest.mark.asyncio
    async def test_analyze_intent_async_error(self) -> None:
        """Test async intent analysis handles errors."""
        with patch("ollama.Client") as mock_client:
            from cue.intelligence import OllamaService
            
            mock_client.return_value.generate.side_effect = Exception("LLM error")
            
            service = OllamaService(
                scratchpad_prompt="{transcription}",
                completion_test_prompt="Complete",
                extraction_prompt="Extract",
            )
            
            async def mock_to_thread(func, *args, **kwargs):
                return func(*args, **kwargs)
            
            with patch("asyncio.to_thread", side_effect=mock_to_thread):
                result = await service.analyze_intent_async("Test input")
                
                assert len(result) == 1
                assert result[0].intent == "Error analyzing intent"
                assert result[0].action == "error"

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
        with patch("ollama.Client"):
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
        with patch("ollama.Client") as mock_ollama:
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

    @pytest.mark.asyncio
    async def test_process_audio_async_success(self) -> None:
        """Test successful async audio processing."""
        with patch("ollama.Client"):
            from cue.intelligence import IntelligenceLayer, TranscriptionResult, IntentResult
            
            layer = IntelligenceLayer(
                deepgram_api_key="test_key",
                deepgram_config={},
                llm_config={
                    "scratchpad_prompt": "{transcription}",
                    "completion_test_prompt": "",
                    "extraction_prompt": "",
                },
            )
            
            # Mock STT
            mock_transcription = TranscriptionResult(
                text="Hello world",
                confidence=0.95,
                duration_seconds=1.0,
                words=[],
            )
            layer.stt.transcribe_async = AsyncMock(return_value=mock_transcription)
            
            # Mock LLM
            mock_intent = IntentResult(
                intent="Greeting",
                action="greet",
                entities={},
                confidence=0.9,
                raw_transcription="Hello world",
                analysis="Greeting detected",
            )
            layer.llm.analyze_intent_async = AsyncMock(return_value=[mock_intent])
            
            result = await layer.process_audio_async(b"audio_data")
            
            assert len(result) == 1
            assert result[0].intent == "Greeting"

    @pytest.mark.asyncio
    async def test_process_audio_async_empty_transcription(self) -> None:
        """Test async audio processing with empty transcription."""
        with patch("ollama.Client"):
            from cue.intelligence import IntelligenceLayer, TranscriptionResult
            
            layer = IntelligenceLayer(
                deepgram_api_key="test_key",
                deepgram_config={},
                llm_config={
                    "scratchpad_prompt": "",
                    "completion_test_prompt": "",
                    "extraction_prompt": "",
                },
            )
            
            # Mock STT with empty result
            mock_transcription = TranscriptionResult(
                text="",
                confidence=0.0,
                duration_seconds=1.0,
                words=[],
            )
            layer.stt.transcribe_async = AsyncMock(return_value=mock_transcription)
            
            result = await layer.process_audio_async(b"audio_data")
            
            assert len(result) == 1
            assert result[0].intent == "No speech detected"

    @pytest.mark.asyncio
    async def test_process_audio_async_with_callback(self) -> None:
        """Test async audio processing invokes callback."""
        with patch("ollama.Client"):
            from cue.intelligence import IntelligenceLayer, TranscriptionResult, IntentResult
            
            layer = IntelligenceLayer(
                deepgram_api_key="test_key",
                deepgram_config={},
                llm_config={
                    "scratchpad_prompt": "{transcription}",
                    "completion_test_prompt": "",
                    "extraction_prompt": "",
                },
            )
            
            mock_transcription = TranscriptionResult(
                text="Test",
                confidence=0.95,
                duration_seconds=1.0,
                words=[],
            )
            layer.stt.transcribe_async = AsyncMock(return_value=mock_transcription)
            
            mock_intent = IntentResult(
                intent="Test",
                action="test",
                entities={},
                confidence=0.9,
                raw_transcription="Test",
                analysis="Test",
            )
            layer.llm.analyze_intent_async = AsyncMock(return_value=[mock_intent])
            
            callback_called = []
            
            def on_transcription(trans):
                callback_called.append(trans)
            
            await layer.process_audio_async(b"audio_data", on_transcription=on_transcription)
            
            assert len(callback_called) == 1
            assert callback_called[0].text == "Test"

    def test_process_audio_sync(self) -> None:
        """Test synchronous process_audio wrapper."""
        with patch("ollama.Client"):
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
            
            with patch.object(layer, "process_audio_async") as mock_async:
                mock_result = MagicMock()
                
                with patch("asyncio.run") as mock_run:
                    mock_run.return_value = mock_result
                    
                    result = layer.process_audio(b"audio_data")
                    
                    mock_run.assert_called_once()


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

    def test_intent_result_defaults(self) -> None:
        """Test IntentResult has correct defaults."""
        from cue.intelligence import IntentResult
        
        result = IntentResult(
            intent="Test",
            action="test",
            entities={},
            confidence=0.5,
            raw_transcription="test",
            analysis="test",
        )
        
        assert result.title == ""
        assert result.category == ""
        assert result.priority == "medium"


# ─────────────────────────────────────────────────────────────────────────────
# TRANSCRIPTION RESULT TESTS
# ─────────────────────────────────────────────────────────────────────────────


class TestTranscriptionResult:
    """Tests for TranscriptionResult data class."""

    def test_transcription_result_creation(self) -> None:
        """Test TranscriptionResult is created correctly."""
        from cue.intelligence import TranscriptionResult
        
        result = TranscriptionResult(
            text="Hello world",
            confidence=0.95,
            duration_seconds=1.5,
            words=[{"word": "Hello"}, {"word": "world"}],
        )
        
        assert result.text == "Hello world"
        assert result.confidence == 0.95
        assert result.duration_seconds == 1.5
        assert len(result.words) == 2

