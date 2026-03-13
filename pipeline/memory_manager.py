# pipeline/memory_manager.py
#
# PURPOSE: Stores conversation history per session in a Python dict (in-memory).
#
# HOW IT FITS:
#   api/chat.py calls get_history(session_id) before the LLM call
#   The history list is passed directly to ollama_client.chat() as messages
#   After the LLM responds, api/chat.py calls add_message() twice:
#   once for "user" role, once for "assistant" role
#
# DATA STRUCTURE:
#   _sessions = {
#     "uuid-abc": [
#       {"role": "user",      "content": "What is P/E?"},
#       {"role": "assistant", "content": "P/E is..."},
#       {"role": "user",      "content": "Give me an example"},
#     ],
#     "uuid-def": [...]
#   }
#
# UPGRADE PATH (post-MVP):
#   Replace _sessions dict with DB reads/writes in future/db.py
#   The function signatures stay the same -- only the storage changes

from collections import defaultdict
from config import MAX_HISTORY
_sessions: dict = defaultdict(list)

def add_message(session_id: str, role: str, content: str) -> None:
    _sessions[session_id].append({"role": role, "content": content})
    _trim(session_id)

def get_history(session_id: str) -> list:
    return _sessions[session_id].copy()

def has_history(session_id: str) -> bool:
    """True if session already has messages. Used by api/chat.py"""
    return len(_sessions[session_id]) > 0

def clear(session_id: str) -> None:
    """Called when user clicks New Chat"""
    _sessions[session_id] = []

def _trim(session_id: str) -> None:
    """Keep last MAX_HISTORY exchanges to avoid context overflow"""
    h = _sessions[session_id]
    if len(h) > MAX_HISTORY * 2:
        _sessions[session_id] = h[-(MAX_HISTORY * 2):]

def trim(session_id: str) -> None:
    """Public alias for _trim. Called by api/chat.py after streaming."""
    _trim(session_id)