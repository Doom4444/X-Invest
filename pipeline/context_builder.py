# pipeline/context_builder.py
#
# Orchestrates all retrieval and assembles the full context block.
# Called by api/chat.py BEFORE the LLM call.
#
# WHAT ALLaM SEES:
#   [Knowledge Base]     ← RAG: hybrid BM25 + semantic (rag/core/retriever.py)
#   [Live Market Data]   ← yfinance via data_fetcher.py
#   [Session Data]       ← online_rag.py for follow-up questions

from rag.core.retriever import Retriever
from pipeline.entity_extractor import extract_tickers
from pipeline.data_fetcher import get_stock_data, format_for_prompt
from pipeline.online_rag import add_to_session, get_session_context

# Instantiate once at module load — avoids rebuilding ChromaDB connection per request
_retriever = Retriever()


def build_context(query: str, session_id: str = "") -> str:
    sections = []

    # 1. Static KB — hybrid BM25 + semantic retrieval from ChromaDB
    try:
        documents, distances = _retriever.retrieve(query, intent="general_finance")
        if documents:
            kb_text = "\n---\n".join(documents)
            sections.append(f"[Knowledge Base]:\n{kb_text}")
    except Exception as e:
        print(f"[ContextBuilder] RAG retrieval failed: {e}")

    # 2. Live data — yfinance for detected tickers
    try:
        tickers = extract_tickers(query)
        live_parts = []
        for ticker in tickers:
            data = get_stock_data(ticker)
            if data:
                fmt = format_for_prompt(data)
                live_parts.append(fmt)
                if session_id:
                    add_to_session(session_id, fmt)
        if live_parts:
            sections.append("[Live Market Data]:\n" + "\n---\n".join(live_parts))
    except Exception as e:
        print(f"[ContextBuilder] Live data fetch failed: {e}")

    # 3. Session data — previously fetched data for follow-up questions
    if session_id:
        session_ctx = get_session_context(session_id)
        if session_ctx and not any("[Live Market Data]" in s for s in sections):
            sections.append(session_ctx)

    return "\n\n".join(sections) if sections else ""