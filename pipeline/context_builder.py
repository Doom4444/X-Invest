# pipeline/context_builder.py
#
# PURPOSE: Orchestrates all retrieval and assembles the full context block.
#
# HOW IT FITS:
#   Called by api/chat.py BEFORE the LLM call
#   Output is passed to ollama_client.chat(context=...)
#   ollama_client prepends it to the last user message
#
# WHAT ALLaM SEES (assembled context block):
#   [Knowledge Base]          <- from rag_retriever (static ChromaDB)
#   P/E Ratio: ...definition
#   ---
#   EPS: ...definition
#
#   [Live Market Data]        <- from data_fetcher (yfinance)
#   Stock: Apple Inc (AAPL)
#   Current Price: 189.30 USD
#   P/E Ratio: 28.5
#
#   [Session Data]            <- from online_rag (this session only)
#   (previous fetched data for follow-up questions)

from rag.retriever import retrieve
from pipeline.entity_extractor import extract_tickers
from pipeline.data_fetcher import get_stock_data, format_for_prompt
from pipeline.online_rag import add_to_session, get_session_context

def build_context(query: str, session_id: str = "") -> str:
    sections = []

    # 1. Static KB (ChromaDB) -- semantic search on financial concepts
    try:
        chunks = retrieve(query, n_results=3)
        if chunks:
            kb_text = "\n---\n".join(c["text"] for c in chunks)
            sections.append(f"[Knowledge Base]:\n{kb_text}")
    except Exception:
        pass

    # 2. Live data -- yfinance for detected tickers
    try:
        tickers = extract_tickers(query)
        live_parts = []
        for ticker in tickers:
            data = get_stock_data(ticker)
            if data:
                fmt = format_for_prompt(data)
                live_parts.append(fmt)
                if session_id:
                    add_to_session(session_id, fmt)  # store for follow-ups
        if live_parts:
            sections.append("[Live Market Data]:\n" + "\n---\n".join(live_parts))
    except Exception:
        pass

    # 3. Session data -- previously fetched data for follow-up questions
    if session_id:
        session_ctx = get_session_context(session_id)
        if session_ctx and not any("[Live Market Data]" in s for s in sections):
            sections.append(session_ctx)

    return "\n\n".join(sections) if sections else ""
