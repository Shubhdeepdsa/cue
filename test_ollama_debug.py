
import asyncio
import ollama

async def test_ollama():
    print("Testing 'llama3.1'...")
    try:
        updated_client = ollama.Client(host='http://localhost:11434')
        response = await asyncio.to_thread(
            updated_client.generate,
            model='llama3.1',
            prompt='Hello',
        )
        print("Success with 'llama3.1'")
    except Exception as e:
        print(f"Failed with 'llama3.1': {e}")

    print("\nTesting 'llama3.1:8b'...")
    try:
        updated_client = ollama.Client(host='http://localhost:11434')
        response = await asyncio.to_thread(
            updated_client.generate,
            model='llama3.1:8b',
            prompt='Hello',
        )
        print("Success with 'llama3.1:8b'")
    except Exception as e:
        print(f"Failed with 'llama3.1:8b': {e}")

if __name__ == "__main__":
    asyncio.run(test_ollama())
