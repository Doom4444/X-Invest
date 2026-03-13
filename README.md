# X-Invest — AI Financial Assistant

> An open-source, educational AI chatbot for stock market research.
> Answers finance questions in **Arabic and English**, pulls **live market data**, and explains financial concepts grounded in a real document knowledge base.

![Python](https://img.shields.io/badge/Python-3.11+-blue?style=flat-square)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-green?style=flat-square)
![Ollama](https://img.shields.io/badge/LLM-ALLaM%207B-orange?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

---

## What is X-Invest?

X-Invest is a vertical AI assistant — meaning it is scoped entirely to the finance domain and will politely refuse questions outside it. It is built as a **graduation project** for Business Information Systems (Egypt, 2026) and designed to run fully locally on consumer hardware.

It is **not** a financial advisor. Every response includes a disclaimer. The goal is education: helping users understand stocks, financial ratios, market signals, and investment concepts.

---

## Features

- **Bilingual** — responds in the language the user writes in (Arabic or English)
- **Live stock data** — detects ticker symbols and fetches real-time prices, P/E, 52-week range, and news via yfinance
- **RAG pipeline** — retrieves relevant chunks from your own document knowledge base before every response
- **Streaming responses** — tokens appear word by word as the model generates them
- **Conversation memory** — maintains per-session history so follow-up questions work naturally
- **Market dashboard** — browse 19 curated companies (US NASDAQ/NYSE, EGX Egypt, Saudi Tadawul) with a live data panel
- **Prediction module** — pluggable signal engine (Bullish / Neutral / Bearish) using technical indicators
- **Finance guardrail** — the system prompt rejects all non-finance queries contextually
- **Disclaimer enforcement** — postprocessor guarantees a disclaimer is always appended

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.11, FastAPI, Uvicorn |
| LLM | ALLaM 7B via Ollama (local, no API key) |
| Embeddings | nomic-embed-text via Ollama |
| Vector Store | ChromaDB (persistent) |
| Live Data | yfinance |
| Frontend | HTML / CSS / Vanilla JS (3 pages, Jinja2) |
| ML (prediction) | scikit-learn, pandas, numpy |

No LangChain. No Docker required. No external AI API calls.

---

## Project Structure

```
X-Invest/
│
├── main.py                     # FastAPI entry point, router registration, page routes
├── config.py                   # All settings loaded from .env — single source of truth
├── requirements.txt
├── .env.example                # Copy to .env and fill in
│
├── api/                        # HTTP endpoints (one file per concern)
│   ├── chat.py                 # POST /api/chat  and  POST /api/chat/stream
│   ├── market.py               # GET  /api/market/companies  and  GET /api/market/{ticker}
│   └── signal.py               # GET  /api/signal/{ticker}
│
├── pipeline/                   # Core request processing pipeline
│   ├── context_builder.py      # Orchestrates RAG + live data → assembles context block
│   ├── memory_manager.py       # Per-session conversation history (in-memory)
│   ├── entity_extractor.py     # Detects stock tickers in Arabic and English text
│   ├── data_fetcher.py         # Fetches and formats yfinance data for the LLM
│   ├── online_rag.py           # Volatile per-session store for fetched live data
│   └── postprocessor.py        # Final cleanup — guarantees disclaimer is present
│
├── rag/                        # Knowledge base pipeline
│   ├── ingest.py               # Reads docs from data/documents/, embeds, stores in ChromaDB
│   └── retriever.py            # MMR semantic search against ChromaDB
│
├── services/
│   └── llm_service.py          # Ollama chat and streaming client (blocking + streaming)
│
├── market/
│   ├── companies.py            # Curated list of 19 tickers (US + EGX + Saudi)
│   └── dashboard.py            # Fetches full company data for the Market UI panel
│
├── prediction/                 # Signal engine — Teammates 2 & 3
│   ├── indicators.py           # Technical indicators (SMA, RSI, MACD, Volatility)
│   ├── model.py                # Random Forest classifier training and inference
│   ├── signal_engine.py        # Public interface: get_signal(ticker) → dict
│   └── saved_models/           # .pkl files (auto-generated, git-ignored)
│
├── models/
│   └── schemas.py              # Pydantic request/response models for all endpoints
│
├── prompts/
│   └── system_prompt.txt       # ALLaM system prompt — edit here to change bot behavior
│
├── data/
│   └── documents/              # Drop PDF, DOCX, TXT, MD files here, then run ingest
│
├── db/
│   └── chroma/                 # ChromaDB persistent store (auto-created, git-ignored)
│
├── templates/                  # Jinja2 HTML pages
│   ├── index.html              # Home / landing page
│   ├── chat.html               # Chat interface
│   └── market.html             # Market browser + live dashboard
│
├── static/
│   ├── css/style.css           # Single stylesheet for all 3 pages
│   ├── js/chat.js              # Streaming chat client, session management
│   └── js/market.js            # Market page — company list + dashboard panel
│
└── future/                     # Scaffolding for post-MVP features (not active)
    ├── auth.py                 # JWT login
    ├── db.py                   # Persistent chat history
    └── upload.py               # File upload RAG
```

---

## How It Works — Full Request Flow

Every chat message goes through this pipeline before a response is generated:

```
User types a message
        │
        ▼
[1] Session check
    Does this session_id have existing history?
    If new: start fresh. If existing: load history from memory_manager.
        │
        ▼
[2] Context Builder  (pipeline/context_builder.py)
    Runs three retrievals in parallel:
    │
    ├── [2a] RAG Retriever  (rag/retriever.py)
    │         Embeds the query via nomic-embed-text (Ollama)
    │         Runs MMR search against ChromaDB
    │         Returns top 3 diverse, relevant document chunks
    │
    ├── [2b] Entity Extractor  (pipeline/entity_extractor.py)
    │         Scans message for ticker patterns (AAPL, TSLA)
    │         and Arabic/English company name aliases
    │         → data_fetcher.py pulls live yfinance data
    │         → online_rag.py stores it for follow-up questions
    │
    └── [2c] Session Store  (pipeline/online_rag.py)
              If no new tickers, retrieves previously fetched
              data from this session (e.g. for follow-ups)
        │
        ▼
[3] Context assembled into one block:
    [Knowledge Base]:         ← from ChromaDB
    P/E Ratio: ...definition
    ---
    [Live Market Data]:       ← from yfinance
    Stock: Apple Inc (AAPL)
    Current Price: 189.30 USD
    ...
        │
        ▼
[4] Memory + LLM call  (services/llm_service.py)
    Prepend context to last user message
    Send full conversation history + system prompt to ALLaM 7B via Ollama
    Stream tokens back as NDJSON
        │
        ▼
[5] Postprocessor  (pipeline/postprocessor.py)
    Check response contains disclaimer keywords
    Append standard disclaimer if missing
        │
        ▼
[6] Streaming response delivered to chat.js
    Tokens rendered word-by-word in the bubble
    On stream end: token count + elapsed time shown below bubble
```

---

## Pages

### `/` — Home
Landing page with a project overview and a "Continue as guest" button that navigates to the chat. Includes a decorative sample portfolio card.

### `/chat` — Chat
The main interface. A fixed input box at the bottom, messages rendered top-down with streaming. Each message has an avatar (`X` for the assistant, `Y` for the user), a bubble that sizes to its content, and a meta line showing token count and response time. "New chat" wipes the session and online RAG data.

### `/market` — Markets
A two-panel layout: a sidebar of 19 curated company cards (US, EGX, Saudi), and a dashboard panel that opens when you click a card. The dashboard shows price, change, P/E, P/B, EPS, dividend yield, 52-week range, day range, volume, and the 5 latest news headlines. The market page works independently — if Ollama is down, this page still loads.

---

## Setup

### Prerequisites

- Python 3.11+
- [Ollama](https://ollama.com) installed and running
- Git

### 1. Clone and install

```bash
git clone https://github.com/your-username/X-Invest.git
cd X-Invest
pip install -r requirements.txt
```

### 2. Pull models

```bash
ollama pull nomic-embed-text:latest
# Choose one of the following chat models:
ollama pull iKhalid/ALLaM:7b     # Arabic-first, recommended
ollama ls
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env if you want to change the model or paths
```

`.env.example`:
```
OLLAMA_URL=http://localhost:11434
MODEL_NAME=iKhalid/ALLaM:7b
EMBED_MODEL=nomic-embed-text:latest
CHROMA_PATH=./db/chroma
COLLECTION_NAME=finance_concepts
NUM_CTX=4096
TEMPERATURE=0.3
MAX_HISTORY=10
PROMPTS_DIR=./prompts
DOCS_PATH=./data/documents
```

### 4. Build the knowledge base (optional but recommended)

Drop any `.pdf`, `.docx`, `.txt`, or `.md` finance documents into `data/documents/`, then:

```bash
python -m rag.ingest
```

This embeds every document and stores chunks in ChromaDB. Re-run whenever you add new documents. Suggested content: financial textbooks, Investopedia exports, EGX guides, CFA notes, Arabic finance references.

### 5. Start the server

```bash
uvicorn main:app --reload
```

Open `http://localhost:8000`.

---

## API Reference

### POST `/api/chat`
Blocking chat endpoint. Returns the full response after generation.

```json
// Request
{ "session_id": "uuid", "message": "What is a P/E ratio?" }

// Response
{ "response": "...", "session_id": "uuid" }
```

### POST `/api/chat/stream`
Streaming endpoint. Returns NDJSON — one token chunk per line.

```json
// Each line during stream:
{"message": {"role": "assistant", "content": "Hello"}, "done": false}

// Final line (Ollama done):
{"done": true, "eval_count": 312, "eval_duration": 4200000000}

// Sentinel line (postprocessed full text):
{"x_invest_final": true, "full_response": "...", "session_id": "uuid"}
```

### POST `/api/clear`
Clears conversation history and session RAG data for a session.

```json
{ "session_id": "uuid" }
```

### GET `/api/market/companies`
Returns the curated company list (19 tickers).

### GET `/api/market/{ticker}`
Returns full dashboard data for a ticker from yfinance.

### GET `/api/signal/{ticker}`
Returns the technical signal for a ticker (Bullish / Neutral / Bearish + confidence). Returns `unavailable` until the prediction module is implemented.

---

## Knowledge Base

The RAG pipeline supports any combination of these file types in `data/documents/`:

| Format | Parser |
|---|---|
| `.pdf` | pdfplumber (handles multi-column layouts) |
| `.docx` | python-docx |
| `.txt` | UTF-8 direct read |
| `.md` | Read + markdown symbols stripped |

Documents are split into 400-word chunks with 50-word overlap, embedded with `nomic-embed-text`, and stored in ChromaDB. Retrieval uses MMR (Maximal Marginal Relevance) to return diverse, relevant chunks — avoiding redundant results.

---

## Prediction Module

The signal engine in `prediction/` is owned by Teammates 2 and 3. The public interface is:

```python
# prediction/signal_engine.py
def get_signal(ticker: str) -> dict:
    return {
        "ticker":     "AAPL",
        "signal":     "bullish",   # bullish | neutral | bearish | unknown
        "confidence": 72.3,        # 0–100
        "rsi":        58.4,
        "sma_cross":  True,
        "rf_signal":  "bullish",
        "disclaimer": "...",
        "error":      ""
    }
```

The function must never raise exceptions. Until implemented, `api/signal.py` returns `"unavailable"` gracefully.

The ML pipeline uses:
- **Features**: SMA crossover, RSI, MACD, Volatility, Volume change
- **Model**: Random Forest classifier
- **Validation**: TimeSeriesSplit (never random split on financial data — that leaks future data)
- **Target**: 5-day forward return bucketed into bullish (>2%) / neutral / bearish (<-2%)

---

## Configuration Reference

All settings live in `.env` and are loaded by `config.py`. No scattered `os.getenv()` calls elsewhere.

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_URL` | `http://localhost:11434` | Ollama server address |
| `MODEL_NAME` | `iKhalid/ALLaM:7b` | Chat model to use |
| `EMBED_MODEL` | `nomic-embed-text:latest` | Embedding model |
| `CHROMA_PATH` | `./db/chroma` | ChromaDB storage location |
| `COLLECTION_NAME` | `finance_concepts` | ChromaDB collection name |
| `NUM_CTX` | `4096` | LLM context window size |
| `TEMPERATURE` | `0.3` | Response randomness (0 = deterministic) |
| `MAX_HISTORY` | `10` | Max conversation turns kept in memory |
| `PROMPTS_DIR` | `./prompts` | Directory containing system_prompt.txt |
| `DOCS_PATH` | `./data/documents` | Directory scanned by `rag.ingest` |

---

## Disclaimer

X-Invest is an educational tool built as a university graduation project. It does not provide professional financial or investment advice. All responses include a disclaimer. Always consult a licensed financial advisor before making investment decisions.

---

## License

MIT — free to use, modify, and build on.
