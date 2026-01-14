"""
CUE Intelligence Module.

Provides the intelligence layer for intent processing:
- Deepgram STT for speech-to-text conversion
- Ollama LLM for intent analysis using scratchpad prompting

Both services are designed for async operation but provide sync wrappers
for simpler integration with the state machine.
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from typing import Any

import ollama


from cue.utils import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class TranscriptionResult:
    """Result from speech-to-text processing."""

    text: str
    confidence: float
    duration_seconds: float
    words: list[dict[str, Any]]


@dataclass
class IntentResult:
    """Result from intent analysis."""

    intent: str
    action: str
    entities: dict[str, Any]
    confidence: float
    raw_transcription: str
    analysis: str


# ─────────────────────────────────────────────────────────────────────────────
# DEEPGRAM SERVICE
# ─────────────────────────────────────────────────────────────────────────────


class DeepgramService:
    """
    Speech-to-text service using Deepgram's REST API directly.
    
    Converts audio bytes to text with high accuracy using
    Deepgram's Nova-2 model via raw HTTP requests.
    
    Attributes:
        api_key: Deepgram API key.
        model: Deepgram model to use.
        language: Target language code.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "nova-2",
        language: str = "en-US",
        smart_format: bool = True,
        punctuate: bool = True,
    ) -> None:
        """
        Initialize the Deepgram service.
        
        Args:
            api_key: Deepgram API key.
            model: Model to use (default: nova-2).
            language: Language code (default: en-US).
            smart_format: Enable smart formatting.
            punctuate: Enable punctuation.
        """
        self.api_key = api_key
        self.model = model
        self.language = language
        self.smart_format = smart_format
        self.punctuate = punctuate
        
        # Base URL for Deepgram API
        self.base_url = "https://api.deepgram.com/v1/listen"
        
        logger.debug(f"Deepgram service initialized (model: {model})")

    async def transcribe_async(self, audio_bytes: bytes) -> TranscriptionResult:
        """
        Transcribe audio bytes to text asynchronously using REST API.
        
        Args:
            audio_bytes: WAV audio data as bytes.
            
        Returns:
            TranscriptionResult with text and metadata.
            
        Raises:
            ValueError: If audio is empty or transcription fails.
        """
        import httpx
        
        if not audio_bytes:
            raise ValueError("Audio bytes cannot be empty")
        
        logger.info(f"Transcribing audio ({len(audio_bytes)} bytes)...")
        
        try:
            # Build query parameters
            params = {
                "model": self.model,
                "language": self.language,
                "smart_format": "true" if self.smart_format else "false",
                "punctuate": "true" if self.punctuate else "false",
            }
            
            headers = {
                "Authorization": f"Token {self.api_key}",
                "Content-Type": "audio/wav",
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.base_url,
                    params=params,
                    headers=headers,
                    content=audio_bytes,
                    timeout=30.0,
                )
                
                if response.status_code != 200:
                    error_msg = f"Deepgram API error: {response.status_code} - {response.text}"
                    logger.error(error_msg)
                    raise ValueError(error_msg)
                
                data = response.json()
                
                # Extract results
                results = data.get("results", {})
                channels = results.get("channels", [])
                
                if not channels:
                    raise ValueError("No transcription channels returned")
                
                alternatives = channels[0].get("alternatives", [])
                if not alternatives:
                    raise ValueError("No transcription alternatives returned")
                
                alternative = alternatives[0]
                text = alternative.get("transcript", "")
                confidence = alternative.get("confidence", 0.0)
                metadata = data.get("metadata", {})
                duration = metadata.get("duration", 0.0)
                
                transcription = TranscriptionResult(
                    text=text,
                    confidence=confidence,
                    duration_seconds=duration,
                    words=[],  # Simplified - skip word-level details
                )
                
                logger.info(f"Transcription complete: '{transcription.text[:50]}...'")
                
                return transcription
            
        except Exception as e:
            logger.error(f"Deepgram transcription failed: {e}")
            raise

    def transcribe(self, audio_bytes: bytes) -> TranscriptionResult:
        """
        Synchronous wrapper for transcribe_async.
        
        Args:
            audio_bytes: WAV audio data as bytes.
            
        Returns:
            TranscriptionResult with text and metadata.
        """
        return asyncio.run(self.transcribe_async(audio_bytes))


# ─────────────────────────────────────────────────────────────────────────────
# OLLAMA SERVICE
# ─────────────────────────────────────────────────────────────────────────────


class OllamaService:
    """
    LLM service using Ollama for local inference.
    
    Uses the "Scratchpad & Completion Test" prompting strategy
    for structured intent analysis.
    
    Attributes:
        model: Ollama model name.
        host: Ollama server host URL.
        timeout: Request timeout in seconds.
    """

    def __init__(
        self,
        model: str = "llama3.1",
        host: str = "http://localhost:11434",
        timeout: int = 30,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        scratchpad_prompt: str = "",
        completion_test_prompt: str = "",
        extraction_prompt: str = "",
    ) -> None:
        """
        Initialize the Ollama service.
        
        Args:
            model: Model name to use.
            host: Ollama server URL.
            timeout: Request timeout.
            temperature: Generation temperature.
            max_tokens: Maximum tokens to generate.
            scratchpad_prompt: Template for scratchpad thinking.
            completion_test_prompt: Template for completion test.
            extraction_prompt: Template for intent extraction.
        """
        self.model = model
        self.host = host
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.scratchpad_prompt = scratchpad_prompt
        self.completion_test_prompt = completion_test_prompt
        self.extraction_prompt = extraction_prompt
        
        self._client = ollama.Client(host=host)
        
        logger.debug(f"Ollama service initialized (model: {model}, host: {host})")

    def _build_scratchpad_prompt(self, transcription: str) -> str:
        """Build the scratchpad prompt with transcription."""
        return self.scratchpad_prompt.format(transcription=transcription)

    def _extract_json(self, text: str) -> dict[str, Any]:
        """
        Extract JSON object from LLM response text.
        
        Args:
            text: Raw LLM response.
            
        Returns:
            Parsed JSON object or list.
        """
        # 1. Try direct parsing (for format="json" mode)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
            
        # 2. Try to find JSON in code blocks
        json_match = re.search(r"```(?:json)?\s*(\[.*?\]|\{.*?\})\s*```", text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # 3. Try to find raw JSON
        try:
            # Look for array first
            start_arr = text.find("[")
            end_arr = text.rfind("]")
            
            # Look for object
            start_obj = text.find("{")
            end_obj = text.rfind("}")
            
            # Use whichever comes first / works best
            if start_arr != -1 and end_arr != -1 and end_arr > start_arr:
                 # Prefer array if it starts before object or if object is inside
                 if start_obj == -1 or start_arr < start_obj:
                    return json.loads(text[start_arr : end_arr + 1])
            
            if start_obj != -1 and end_obj != -1 and end_obj > start_obj:
                return json.loads(text[start_obj : end_obj + 1])
                
        except json.JSONDecodeError:
            pass
        
        # Default fallback
        logger.warning(f"Failed to extract JSON from: {text[:100]}...")
        return [{
            "intent": "Unknown",
            "action": "unknown",
            "entities": {},
            "confidence": 0.5,
        }]

    async def analyze_intent_async(self, transcription: str) -> list[IntentResult]:
        """
        Analyze transcription to extract intent using scratchpad method.
        
        Args:
            transcription: The transcribed speech text.
            
        Returns:
            List of IntentResult objects.
        """
        if not transcription.strip():
            return [IntentResult(
                intent="Empty input",
                action="none",
                entities={},
                confidence=0.0,
                raw_transcription=transcription,
                analysis="No transcription provided",
            )]
        
        logger.info("Analyzing intent with Ollama...")
        
        try:
            # Step 1: Scratchpad thinking
            scratchpad_prompt = self._build_scratchpad_prompt(transcription)
            
            scratchpad_response = await asyncio.to_thread(
                self._client.generate,
                model=self.model,
                prompt=scratchpad_prompt,
                options={
                    "temperature": self.temperature,
                    "num_predict": self.max_tokens,
                },
            )
            
            analysis = scratchpad_response.get("response", "")
            logger.debug(f"Scratchpad analysis: {analysis[:100]}...")
            
            # Step 2: Completion test
            completion_prompt = f"{scratchpad_prompt}\n\n{analysis}\n\n{self.completion_test_prompt}"
            
            completion_response = await asyncio.to_thread(
                self._client.generate,
                model=self.model,
                prompt=completion_prompt,
                options={
                    "temperature": 0.3,  # Lower temp for structured output
                    "num_predict": 256,
                },
            )
            
            completion_result = completion_response.get("response", "")
            logger.debug(f"Completion test: {completion_result[:100]}...")
            
            # Step 3: Intent extraction
            extraction_prompt = f"{completion_prompt}\n\n{completion_result}\n\n{self.extraction_prompt}"
            
            extraction_response = await asyncio.to_thread(
                self._client.generate,
                model=self.model,
                prompt=extraction_prompt,
                format="json",
                options={
                    "temperature": 0.1,  # Very low for JSON output
                    "num_predict": 256,
                },
            )
            
            extraction_result = extraction_response.get("response", "")
            logger.debug(f"Extraction result: {extraction_result}")
            
            # Parse the JSON result
            intent_data = self._extract_json(extraction_result)

            # Handle list directly (if model returns array)
            if isinstance(intent_data, list):
                results_data = intent_data
            # Handle wrapped object (if model returns {"intents": [...]})
            elif isinstance(intent_data, dict) and "intents" in intent_data and isinstance(intent_data["intents"], list):
                results_data = intent_data["intents"]
            # Handle single object (fallback)
            elif isinstance(intent_data, dict):
                results_data = [intent_data]
            else:
                results_data = []

            results = []
            for item in results_data:
                results.append(IntentResult(
                    intent=item.get("intent", "Unknown"),
                    action=item.get("action", "unknown"),
                    entities=item.get("entities", {}),
                    confidence=float(item.get("confidence", 0.5)),
                    raw_transcription=transcription,
                    analysis=analysis,
                ))
            
            logger.info(f"Extracted {len(results)} intents")
            
            return results
            
        except Exception as e:
            logger.error(f"Ollama analysis failed: {e}")
            return [IntentResult(
                intent="Error analyzing intent",
                action="error",
                entities={},
                confidence=0.0,
                raw_transcription=transcription,
                analysis=str(e),
            )]

    def analyze_intent(self, transcription: str) -> list[IntentResult]:
        """
        Synchronous wrapper for analyze_intent_async.
        
        Args:
            transcription: The transcribed speech text.
            
        Returns:
            List of IntentResult objects.
        """
        return asyncio.run(self.analyze_intent_async(transcription))

    def health_check(self) -> bool:
        """
        Check if Ollama server is available.
        
        Returns:
            True if server is healthy, False otherwise.
        """
        try:
            # Try to list models
            self._client.list()
            return True
        except Exception as e:
            logger.warning(f"Ollama health check failed: {e}")
            return False


# ─────────────────────────────────────────────────────────────────────────────
# INTELLIGENCE LAYER
# ─────────────────────────────────────────────────────────────────────────────


class IntelligenceLayer:
    """
    Orchestrates the complete intelligence pipeline.
    
    Combines Deepgram STT and Ollama LLM to provide
    end-to-end speech-to-intent processing.
    """

    def __init__(
        self,
        deepgram_api_key: str,
        deepgram_config: dict,
        llm_config: dict,
    ) -> None:
        """
        Initialize the intelligence layer.
        
        Args:
            deepgram_api_key: API key for Deepgram.
            deepgram_config: Deepgram configuration dict.
            llm_config: LLM configuration dict.
        """
        self.stt = DeepgramService(
            api_key=deepgram_api_key,
            model=deepgram_config.get("model", "nova-2"),
            language=deepgram_config.get("language", "en-US"),
            smart_format=deepgram_config.get("smart_format", True),
            punctuate=deepgram_config.get("punctuate", True),
        )
        
        self.llm = OllamaService(
            model=llm_config.get("model", "llama3.1"),
            host=llm_config.get("host", "http://localhost:11434"),
            timeout=llm_config.get("timeout_seconds", 30),
            temperature=llm_config.get("temperature", 0.7),
            max_tokens=llm_config.get("max_tokens", 1024),
            scratchpad_prompt=llm_config.get("scratchpad_prompt", ""),
            completion_test_prompt=llm_config.get("completion_test_prompt", ""),
            extraction_prompt=llm_config.get("extraction_prompt", ""),
        )
        
        logger.info("Intelligence layer initialized")

    async def process_audio_async(
        self,
        audio_bytes: bytes,
        on_transcription: callable | None = None,
    ) -> IntentResult:
        """
        Process audio through the complete pipeline.
        
        Args:
            audio_bytes: WAV audio data.
            on_transcription: Optional callback after transcription.
            
        Returns:
            List of IntentResults from the complete pipeline.
        """
        # Step 1: Speech-to-text
        transcription = await self.stt.transcribe_async(audio_bytes)
        
        if on_transcription:
            on_transcription(transcription)
        
        if not transcription.text.strip():
            logger.warning("Empty transcription")
            return [IntentResult(
                intent="No speech detected",
                action="none",
                entities={},
                confidence=0.0,
                raw_transcription="",
                analysis="",
            )]
        
        # Step 2: Intent analysis
        intents = await self.llm.analyze_intent_async(transcription.text)
        
        return intents

    def process_audio(
        self,
        audio_bytes: bytes,
        on_transcription: callable | None = None,
    ) -> IntentResult:
        """
        Synchronous wrapper for process_audio_async.
        
        Args:
            audio_bytes: WAV audio data.
            on_transcription: Optional callback after transcription.
            
        Returns:
            List of IntentResults from the complete pipeline.
        """
        return asyncio.run(self.process_audio_async(audio_bytes, on_transcription))

    def health_check(self) -> dict[str, bool]:
        """
        Check health of all intelligence services.
        
        Returns:
            Dict with service health status.
        """
        return {
            "ollama": self.llm.health_check(),
            # Deepgram doesn't have a simple health check
            "deepgram": bool(self.stt.api_key),
        }
