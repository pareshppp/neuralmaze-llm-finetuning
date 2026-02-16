import os
import sys
from openai import OpenAI
import fire

def stream_base_model(
    prompt: str,
    model: str,
    api_url: str = None,
    api_key: str = None,
    max_tokens: int = 1024,
    temperature: float = 0.7,
    stop_sequences: list = None
):
    # 1. Handle URL formatting
    # Hugging Face Endpoints usually want the base URL ending in /v1/
    url = api_url or os.getenv("HF_ENDPOINT_URL")
    key = api_key or os.getenv("HF_API_TOKEN")

    if not url or not key:
        print("Error: Missing HF_ENDPOINT_URL or HF_API_TOKEN")
        return

    # Ensure URL ends with /v1/ for the OpenAI SDK
    if not url.endswith("/v1/"):
        url = url.rstrip("/") + "/v1/"

    client = OpenAI(base_url=url, api_key=key)

    if stop_sequences is None:
        stop_sequences = ["\n\n", "User:"]

    try:
        # Using completions.create for Base Models
        stream = client.completions.create(
            model=model, 
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop_sequences,
            stream=True
        )

        for chunk in stream:
            # For .completions, the text is in chunk.choices[0].text
            if chunk.choices and chunk.choices[0].text:
                print(chunk.choices[0].text, end="", flush=True)
                
    except Exception as e:
        print(f"\n\n[Connection Error]: {e}")
        print(f"Attempted URL: {url}")

if __name__ == "__main__":
    fire.Fire(stream_base_model)