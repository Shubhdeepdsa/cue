
import asyncio
import logging
import sys

# Configure logging to show debug output
logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

from cue.intelligence import OllamaService
from cue.config import settings

async def debug_intent():
    config = settings
    
    # Initialize service with config
    service = OllamaService(
        model=config.llm.model,
        host=config.llm.host,
        timeout=config.llm.timeout_seconds,
        temperature=config.llm.temperature,
        max_tokens=config.llm.max_tokens,
        scratchpad_prompt=config.llm.scratchpad_prompt,
        completion_test_prompt=config.llm.completion_test_prompt,
        extraction_prompt=config.llm.extraction_prompt,
    )
    
    test_input = "I want to start deployment for the front end application, pick up my mom, and I have to fix the leaking pipe in my building."
    
    print(f"\nAnalyzing input: '{test_input}'\n")
    try:
        result = await service.analyze_intent_async(test_input)
        
        output = f"""
=== Result ===
Intent: {result.intent}
Action: {result.action}
Confidence: {result.confidence}
Analysis: {result.analysis}
Raw Transcription: {result.raw_transcription}
Entities: {result.entities}
"""
        print(output)
        with open("debug_output.txt", "w") as f:
            f.write(output)
            
    except Exception as e:
        error_msg = f"Error: {e}"
        print(error_msg)
        with open("debug_output.txt", "w") as f:
            f.write(error_msg)

if __name__ == "__main__":
    asyncio.run(debug_intent())
