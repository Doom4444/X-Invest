# api/chat.py
#
# PURPOSE: The /api/chat endpoint -- core of the application.
#
# HOW IT FITS:
#   Registered in main.py via app.include_router(chat_router)
#   Called by static/js/chat.js via fetch POST /api/chat (non-streaming)
#   Called by static/js/chat.js via fetch POST /api/chat/stream (streaming)
#   Orchestrates the full pipeline:
#     context_builder -> memory -> ollama_client -> postprocessor
#
# TWO ENDPOINTS:
#   POST /api/chat         → blocking, returns full JSON (keep for fallback/testing)
#   POST /api/chat/stream  → streaming NDJSON, tokens arrive in real time
#
# REQUEST (both):  { "session_id": "uuid", "message": "What is P/E?" }
# RESPONSE /chat:        { "response": "...", "session_id": "uuid" }
# RESPONSE /chat/stream: NDJSON stream, one Ollama chunk per line
#
# NO GUARDRAIL HERE -- the system prompt in ollama_client.py handles domain
# filtering contextually. The old keyword guardrail blocked follow-ups.

import json
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pipeline.memory_manager import add_message, get_history, clear, trim
from pipeline.context_builder import build_context
from services.llm_service import chat, stream_chat
from pipeline.postprocessor import process
from pipeline.online_rag import clear_session

router = APIRouter()


@router.post("/api/chat")
async def chat_endpoint(request: Request):
    """
    Blocking endpoint — waits for the full response before returning.
    Keep this for:
      - API testing via /docs
      - Any future programmatic callers that need the full text at once
    chat.js uses /api/chat/stream instead.
    """
    body       = await request.json()
    session_id = body.get("session_id", "default")
    user_query = body.get("message", "").strip()

    if not user_query:
        return JSONResponse({"error": "Empty message"}, status_code=400)

    # Step 1: Build context BEFORE adding to history
    # (pure query gives better retrieval than query+history noise)
    context = build_context(user_query, session_id=session_id)

    # Step 2: Add to memory and get full history
    add_message(session_id, "user", user_query)
    history = get_history(session_id)

    # Step 3: Call LLM with history (memory) + context (grounding)
    response = chat(history, context=context)

    # Step 4: Save response to memory
    add_message(session_id, "assistant", response)

    # Step 5: Post-process and return
    return JSONResponse({"response": process(response), "session_id": session_id})


@router.post("/api/chat/stream")
async def chat_stream_endpoint(request: Request):
    """
    Streaming endpoint — yields tokens as Ollama produces them.
    chat.js connects here and appends each token to the bubble in real time.

    STREAM FORMAT (NDJSON — one JSON object per line):
        {"message": {"role": "assistant", "content": "Hello"}, "done": false}
        {"message": {"role": "assistant", "content": " world"}, "done": false}
        {"done": true, "eval_count": 312, "eval_duration": 4200000000}

    MEMORY STRATEGY:
        We collect the full response while streaming, then save it to memory
        and run postprocessor AFTER the stream ends via a sentinel final chunk.
        The sentinel is a custom NDJSON line with "x_invest_final": true
        and the post-processed text, so chat.js can store it locally if needed.
        Memory is saved server-side regardless.

    ERROR HANDLING:
        If stream_chat() hits an error it yields a JSON line with "error" key.
        chat.js checks for this and displays the error in the bubble.
    """
    body       = await request.json()
    session_id = body.get("session_id", "default")
    user_query = body.get("message", "").strip()

    if not user_query:
        return JSONResponse({"error": "Empty message"}, status_code=400)

    # Build context and update memory BEFORE opening the stream
    # (same order as the blocking endpoint — keeps behavior consistent)
    context = build_context(user_query, session_id=session_id)
    add_message(session_id, "user", user_query)
    history = get_history(session_id)

    # Accumulate tokens here so we can save to memory after stream ends
    full_response_tokens: list[str] = []

    def generate():
        for chunk in stream_chat(history, context=context):
            if not chunk:
                continue

            yield chunk

            # Parse each chunk to collect tokens and detect done
            # chunk is raw bytes — may contain multiple lines if Ollama batched them
            try:
                for line in chunk.decode("utf-8", errors="ignore").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)

                    # Accumulate content tokens
                    token = data.get("message", {}).get("content", "")
                    if token:
                        full_response_tokens.append(token)

                    # When Ollama signals done, save to memory + postprocess
                    if data.get("done"):
                        full_text     = "".join(full_response_tokens)
                        final_text    = process(full_text)
                        add_message(session_id, "assistant", final_text)
                        trim(session_id)

                        # Send one extra sentinel line so chat.js knows the
                        # cleaned/disclaimer-appended version of the full text
                        sentinel = json.dumps({
                            "x_invest_final": True,
                            "full_response":  final_text,
                            "session_id":     session_id,
                        }) + "\n"
                        yield sentinel.encode("utf-8")

            except (json.JSONDecodeError, UnicodeDecodeError):
                # Malformed chunk — skip, don't crash the stream
                continue

    return StreamingResponse(
        generate(),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control":    "no-cache",
            "X-Accel-Buffering": "no",   # disables nginx buffering if behind a proxy
        }
    )


@router.post("/api/clear")
async def clear_endpoint(request: Request):
    """Called by New Chat button in the frontend."""
    body = await request.json()
    sid  = body.get("session_id", "default")
    clear(sid)          # wipe conversation history
    clear_session(sid)  # wipe volatile online RAG data
    return JSONResponse({"status": "cleared"})