# services/llm_service.py

import json
import requests
import os
from typing import Generator
from config import OLLAMA_URL, MODEL_NAME, NUM_CTX, TEMPERATURE, PROMPTS_DIR


def _load_system_prompt() -> str:
    """Load system prompt from prompts/system_prompt.txt."""
    path = os.path.join(PROMPTS_DIR, "system_prompt.txt")
    try:
        with open(path, encoding="utf-8") as f:
            return f.read().strip()
    except FileNotFoundError:
        print(f"[ollama_client] Warning: {path} not found. Using fallback prompt.")
        return (
            "You are X-Invest, an educational financial assistant. "
            "Only answer finance-related questions. "
            "Always add a disclaimer that this is not professional financial advice."
        )


# Load once at startup — not on every request
SYSTEM_PROMPT = _load_system_prompt()


def _enrich(messages: list, context: str) -> list:
    """
    Prepend RAG/live-data context to the last user message.
    Used by both chat() and stream_chat() so logic stays in one place.
    """
    enriched = messages.copy()
    if context and enriched:
        enriched[-1]["content"] = context + "\n\n" + enriched[-1]["content"]
    return enriched


def chat(messages: list, context: str = "") -> str:
    """
    Blocking (non-streaming) call to Ollama.
    Returns the full response string.
    Used by any code path that needs the complete answer before continuing.
    """
    try:
        r = requests.post(f"{OLLAMA_URL}/api/chat", json={
            "model":    MODEL_NAME,
            "system":   SYSTEM_PROMPT,
            "messages": _enrich(messages, context),
            "stream":   False,
            "options":  {"temperature": TEMPERATURE, "num_ctx": NUM_CTX}
        }, timeout=120)
        r.raise_for_status()
        return r.json()["message"]["content"].strip()
    except requests.exceptions.ConnectionError:
        return "Error: Cannot connect to Ollama. Run: ollama serve"
    except requests.exceptions.Timeout:
        return "Error: Model timed out. Please try again."
    except Exception as e:
        return f"Unexpected error: {str(e)}"


def stream_chat(messages: list, context: str = "") -> Generator[bytes, None, None]:
    """
    Streaming call to Ollama. Yields raw NDJSON bytes chunk by chunk.

    HOW IT WORKS:
        Ollama streams the response as one JSON object per line (NDJSON).
        Each line looks like:
            {"message": {"role": "assistant", "content": "Hello"}, "done": false}
        The final line has "done": true and includes token stats:
            {"done": true, "eval_count": 312, "eval_duration": 4200000000}

    HOW THE CALLER USES IT:
        api/chat.py wraps this in a FastAPI StreamingResponse.
        chat.js reads the stream with a ReadableStream reader,
        splits on newlines, parses each JSON line, and appends
        data.message.content to the bubble token by token.

    ERROR HANDLING:
        If Ollama is unreachable or crashes mid-stream, we yield a
        final JSON line with an "error" key so chat.js can display it
        cleanly instead of the stream just going silent.
    """
    try:
        r = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json={
                "model":    MODEL_NAME,
                "system":   SYSTEM_PROMPT,
                "messages": _enrich(messages, context),
                "stream":   True,
                "options":  {"temperature": TEMPERATURE, "num_ctx": NUM_CTX}
            },
            stream=True,
            timeout=120
        )
        r.raise_for_status()

        for chunk in r.iter_content(chunk_size=None):
            if chunk:
                yield chunk

    except requests.exceptions.ConnectionError:
        yield json.dumps({
            "error": "Cannot connect to Ollama. Run: ollama serve",
            "done": True
        }).encode("utf-8") + b"\n"

    except requests.exceptions.Timeout:
        yield json.dumps({
            "error": "Model timed out. Please try again.",
            "done": True
        }).encode("utf-8") + b"\n"

    except Exception as e:
        yield json.dumps({
            "error": f"Unexpected error: {str(e)}",
            "done": True
        }).encode("utf-8") + b"\n"