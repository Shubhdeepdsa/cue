
import asyncio
import logging
import sys
from cue.intelligence import OllamaService
from cue.config import settings

# Subclass to spy on internals
class DebugOllamaService(OllamaService):
    async def analyze_intent_async(self, transcription):
        print(f"DEBUG: Analying '{transcription}'")
        
        if not transcription.strip():
            return []
            
        # Step 1: Scratchpad
        scratchpad_prompt = self._build_scratchpad_prompt(transcription)
        print("DEBUG: Sending scratchpad request...")
        resp1 = await asyncio.to_thread(self._client.generate, model=self.model, prompt=scratchpad_prompt, options={"num_predict": self.max_tokens})
        analysis = resp1.get("response", "")
        print(f"DEBUG: Scratchpad output len: {len(analysis)}")
        
        # Step 2: Completion
        completion_prompt = f"{scratchpad_prompt}\n\n{analysis}\n\n{self.completion_test_prompt}"
        print("DEBUG: Sending completion request...")
        resp2 = await asyncio.to_thread(self._client.generate, model=self.model, prompt=completion_prompt, options={"num_predict": 256})
        completion_result = resp2.get("response", "")
        print(f"DEBUG: Completion result:\n{completion_result}\n")
        print(f"DEBUG: Completion output len: {len(completion_result)}")
        
        # Step 3: Extraction
        extraction_prompt = f"{completion_prompt}\n\n{completion_result}\n\n{self.extraction_prompt}"
        print("DEBUG: Sending extraction request (format='json')...")
        resp3 = await asyncio.to_thread(self._client.generate, model=self.model, prompt=extraction_prompt, format="json", options={"temperature": 0.1, "num_predict": 256})
        extraction_result = resp3.get("response", "")
        
        print("\n\n=== RAW EXTRACTION RESULT START ===")
        print(extraction_result)
        print("=== RAW EXTRACTION RESULT END ===\n\n")
        
        # Use underlying logic to parse
        results = await super().analyze_intent_async(transcription)
        print(f"\nparsed {len(results)} results")
        return results

async def run_debug():
    config = settings
    service = DebugOllamaService(
        model=config.llm.model,
        host=config.llm.host,
        timeout=config.llm.timeout_seconds,
        temperature=config.llm.temperature,
        max_tokens=config.llm.max_tokens,
        scratchpad_prompt=config.llm.scratchpad_prompt,
        completion_test_prompt=config.llm.completion_test_prompt,
        extraction_prompt=config.llm.extraction_prompt,
    )
    
    input_text = "I want to start deployment for the front end application, pick up my mom, and I have to fix the leaking pipe in my building."
    results = await service.analyze_intent_async(input_text)
    
    for i, res in enumerate(results):
        print(f"\n--- Intent {i+1} ---")
        print(f"Intent: {res.intent}")
        print(f"Action: {res.action}")
        print(f"Confidence: {res.confidence}")

if __name__ == "__main__":
    asyncio.run(run_debug())
