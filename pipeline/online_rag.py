# pipeline/online_rag.py
#
# PURPOSE: Volatile per-session data store for real-time fetched content.
#
# HOW IT FITS:
#   When context_builder.py fetches live data for a ticker, it also
#   stores it here via add_to_session(session_id, text)
#   On follow-up questions ("what do you think of its price?"), the
#   session data is retrieved and included in context even without new tickers
#   When user clicks New Chat, api/chat.py calls clear_session()
#
# WHY SEPARATE FROM STATIC RAG:
#   Static RAG (rag_retriever.py) = persistent knowledge base (concepts.json)
#   Online RAG (this file) = fresh data scoped to one conversation session
#   Mixing them would pollute your persistent KB with today's stock prices

from collections import defaultdict

_session_data: dict[str, list[str]] = defaultdict(list)

def add_to_session(session_id: str, text: str) -> None:
    if text and text not in _session_data[session_id]:
        _session_data[session_id].append(text)

def get_session_context(session_id: str) -> str:
    chunks = _session_data.get(session_id, [])
    if not chunks: return ""
    return "[Session Live Data]:\n" + "\n---\n".join(chunks)

def clear_session(session_id: str) -> None:
    _session_data[session_id] = []
