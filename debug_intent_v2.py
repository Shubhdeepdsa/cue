
import asyncio
import logging
import sys
from cue.intelligence import OllamaService
from cue.config import settings
from cue.utils import get_logger

# Subclass to spy on internals
class DebugOllamaService(OllamaService):
    async def analyze_intent_async(self, transcription):
        print(f"DEBUG: Analying '{transcription}'")
        
        if not transcription.strip():
            return []
            
        # Step 1: Scratchpad
        scratchpad_prompt = self.config["scratchpad_prompt"].format(transcription=transcription)
        print("DEBUG: Sending scratchpad request...")
        resp1 = await asyncio.to_thread(
            self._client.generate, 
            model=self.model, 
            prompt=scratchpad_prompt, 
            options={"num_predict": 1024, "temperature": 0.7}
        )
        analysis = resp1.get("response", "")
        print(f"DEBUG: Scratchpad output len: {len(analysis)}")
        print(f"\n--- SCRATCHPAD ANALYSIS ---\n{analysis}\n---------------------------\n")
        
        # Step 2: Extraction
        extraction_prompt = self.config["extraction_prompt"] + f"\n\nAnalysis Context:\n{analysis}"
        print("DEBUG: Sending extraction request (format='json')...")
        resp2 = await asyncio.to_thread(
            self._client.generate, 
            model=self.model, 
            prompt=extraction_prompt, 
            format="json", 
            options={"temperature": 0.1, "num_predict": 512}
        )
        extraction_result = resp2.get("response", "")
        
        print("\n\n=== RAW EXTRACTION RESULT START ===")
        print(extraction_result)
        print("=== RAW EXTRACTION RESULT END ===\n\n")
        
        # Parse logic
        raw_data = self._extract_json(extraction_result)
        
        tickets_data = []
        if isinstance(raw_data, dict) and "tickets" in raw_data and isinstance(raw_data["tickets"], list):
            tickets_data = raw_data["tickets"]
        elif isinstance(raw_data, list):
            tickets_data = raw_data
        elif isinstance(raw_data, dict):
             if "title" in raw_data or "intent" in raw_data:
                 tickets_data = [raw_data]

        from cue.intelligence import IntentResult
        results = []
        for item in tickets_data:
            results.append(IntentResult(
                title=item.get("title", item.get("intent", "Unknown Task")),
                category=item.get("category", "Uncategorized"),
                next_action=item.get("next_action", item.get("action", "Review")),
                estimated_time=item.get("estimated_time", "Unknown"),
                priority=item.get("priority", "Medium"),
                confidence=float(item.get("confidence", 0.5)),
                raw_transcription=transcription,
                analysis=analysis,
            ))
            
        print(f"\nParsed {len(results)} results")
        return results

async def run_debug():
    # Force load config
    config = settings.llm.model_dump()
    # Add prompts explicitly if needed, but model_dump should have them
    # Actually settings.llm is a Pydantic model, model_dump converts to dict
    
    # We need to manually inject the new prompts if they weren't in the model definition?
    # Wait, Config is loaded from yaml. If I updated yaml, settings should pick it up 
    # IF the pydantic model has those fields.
    # The pydantic model for LLMConfig likely has `scratchpad_prompt`, `completion_test_prompt`, `extraction_prompt`.
    # I updated config.yaml. The Pydantic model checks env/yaml.
    # I assume it works.
    
    service = DebugOllamaService(config)
    
    input_text = "I want to start deployment for the front end application, pick up my mom, and I have to fix the leaking pipe in my building."
    results = await service.analyze_intent_async(input_text)
    
    for i, res in enumerate(results):
        print(f"\n--- Ticket {i+1} ---")
        print(f"Title: {res.title}")
        print(f"Category: {res.category}")
        print(f"Action: {res.next_action}")
        print(f"Time: {res.estimated_time}")
        print(f"Priority: {res.priority}")
        print(f"Confidence: {res.confidence}")

if __name__ == "__main__":
    asyncio.run(run_debug())
