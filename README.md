# X-Invest — AI Financial Assistant & Backtest Simulator

> An open-source, educational AI chatbot and trading strategy backtest simulator for stock market research.
> Answers finance questions in **Arabic and English**, retrieves **live market data**, explains financial concepts grounded in a document knowledge base, and simulates algorithmic trading strategies.

![Python](https://img.shields.io/badge/Python-3.12+-blue?style=flat-square)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-green?style=flat-square)
![Ollama](https://img.shields.io/badge/LLM-ALLaM%207B-orange?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-lightgrey?style=flat-square)

---

## What is X-Invest?

X-Invest is a vertical AI assistant scoped entirely to the finance domain (it contextually refuses non-finance queries). Built as a university graduation project, it is designed to run fully locally on consumer hardware.

It is **not** a financial advisor (disclaimers are automatically enforced in all responses). The goal is education: helping users understand stocks, financial metrics, RAG-grounded investment concepts, and simulated trading performance.

---

## Key Features

- **Bilingual Chat** — responds in the language of the user's query (Arabic or English).
- **Live Stock Data** — automatically extracts stock tickers and pulls live prices, P/E, 52-week ranges, and news via `yfinance`.
- **RAG Pipeline** — hybrid BM25 + semantic search over your local document knowledge base (CFA guides, books, etc.) using ChromaDB.
- **Markets Dashboard** — browse 19 curated US and Middle Eastern companies (NASDAQ, NYSE, EGX, Tadawul) with an instant-load live data panel.
- **Preloading Caching Layer** — parallel macro downloads and launch-time pre-caching for fast Markets dashboard loads.
- **Backtest Simulator** — test strategy performance on historical data with equity charts, drawdown calculations, and a trade ledger.
- **ML Signal Engine** — Random Forest + XGBoost ensemble predicting 5-day forward return directions (Bullish, Bearish, or Neutral).

---

## Quick Start

Get the web app running in a few minutes. Choose a setup profile below if you need more than the defaults.

### 1. Prerequisites

| Requirement | Notes |
|---|---|
| **Python 3.12+** | [python.org](https://www.python.org/downloads/) |
| **Git** | To clone the repository |
| **Ollama** | [ollama.com](https://ollama.com) — runs the LLM and embeddings locally |

### 2. Clone and install

```bash
git clone https://github.com/your-username/X-Invest.git
cd X-Invest

# Create a virtual environment (pick one)
python -m venv .venv                        # venv (all platforms)
# conda create -n xinvest python=3.12 -y    # Conda alternative

# Activate it
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Windows (cmd):
.venv\Scripts\activate.bat
# macOS / Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Pull Ollama models

Start the Ollama app, then pull the models referenced in `.env.example`:

```bash
ollama pull iKhalid/ALLaM:7b
ollama pull bge-m3:latest
```

### 4. Configure environment

```bash
# Windows
copy .env.example .env

# macOS / Linux
cp .env.example .env
```

The defaults work out of the box for a local Ollama install. See [Configuration](#configuration) below to customize.

### 5. Run the app

```bash
uvicorn main:app --reload
```

Open **http://localhost:8000**. On first startup, the server pre-warms market data (~10 seconds) before showing `Application startup complete`.

| Page | URL |
|---|---|
| Home | http://localhost:8000 |
| Chat | http://localhost:8000/chat |
| Markets | http://localhost:8000/market |
| Backtest | http://localhost:8000/backtest |
| API docs | http://localhost:8000/docs |

---

## Setup Profiles

Pick the profile that matches what you want to run. All profiles share the same install steps above; only the optional steps and `.env` values differ.

### Profile A — Chat & Markets only (fastest)

Use this to try the assistant and dashboard without training ML models or ingesting documents.

1. Complete [Quick Start](#quick-start) steps 1–5.
2. Skip RAG ingest and model training.

**Works:** bilingual chat, live yfinance data, Markets dashboard.  
**Limited:** no document-grounded answers (empty knowledge base), no ML signals or web backtest until models are trained.

### Profile B — Full local (recommended)

Everything runs offline on your machine with no paid API keys.

1. Complete [Quick Start](#quick-start).
2. Add finance documents to `data/documents/` (`.pdf`, `.docx`, `.txt`, `.md`).
3. Ingest the knowledge base:
   ```bash
   python -m rag.preprocessing.ingest
   ```
4. Train prediction models (required for Backtest page and chat ML signals):
   ```bash
   python prediction/train.py
   ```
5. Start the server: `uvicorn main:app --reload`

### Profile C — Remote Ollama server

Run Ollama on another machine (e.g. a GPU server) while the app runs locally.

Edit `.env`:

```env
OLLAMA_URL=http://192.168.1.50:11434
MODEL_NAME=iKhalid/ALLaM:7b
EMBED_MODEL=bge-m3:latest
```

Pull models on the **remote** host, not your laptop. Ensure port `11434` is reachable from the machine running X-Invest.

### Profile D — Alternative LLM or embedding model

Swap models without code changes — set names to any model you have pulled in Ollama:

```env
MODEL_NAME=llama3.2:latest
EMBED_MODEL=bge-m3:latest
```

After changing models, re-run ingest if you use RAG (`python -m rag.preprocessing.ingest`), because embeddings must match `EMBED_MODEL`.

```bash
ollama pull llama3.2:latest
ollama pull bge-m3:latest
```

### Profile E — Low-memory / slower hardware

Reduce context size and conversation history:

```env
NUM_CTX=2048
MAX_HISTORY=6
TEMPERATURE=0.2
```

Use a smaller chat model if ALLaM 7B is too heavy for your GPU/RAM.

### Profile F — Optional external APIs

Core features use **yfinance** and **Ollama** only. These keys unlock extra news and sentiment paths; leave them blank to skip.

| Variable | Used by | Get a key |
|---|---|---|
| `FINNHUB_API_KEY` | `rag/online/news_fetcher.py`, `market_fetcher.py` | [finnhub.io](https://finnhub.io) |
| `TWELVEDATA_API_KEY` | `rag/online/market_fetcher.py` | [twelvedata.com](https://twelvedata.com) |
| `NEWS_API_KEY` | `prediction/Sentiment.py` | [newsapi.org](https://newsapi.org) |
| `HF_TOKEN` | Hugging Face model downloads (FinBERT / transformers) | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) |

Add only the keys you need to `.env`:

```env
FINNHUB_API_KEY=your_key_here
TWELVEDATA_API_KEY=your_key_here
NEWS_API_KEY=your_key_here
HF_TOKEN=your_token_here
```

---

## Configuration

All settings are loaded from `.env` by `config.py`. Restart the server after any change.

### Core settings (in `config.py`)

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_URL` | `http://localhost:11434` | Ollama API base URL |
| `MODEL_NAME` | `iKhalid/ALLaM:7b` | Chat / completion model |
| `EMBED_MODEL` | `bge-m3:latest` | Embedding model for ChromaDB |
| `NUM_CTX` | `4096` | Ollama context window (tokens) |
| `TEMPERATURE` | `0.3` | Sampling temperature (0 = deterministic) |
| `MAX_HISTORY` | `10` | Max user+assistant turn pairs kept in memory |
| `CHROMA_PATH` | `./db/chroma` | ChromaDB storage directory |
| `COLLECTION_NAME` | `finance_concepts` | Chroma collection name |
| `PROMPTS_DIR` | `./prompts` | System prompt directory |
| `DOCS_PATH` | `./data/documents` | Folder scanned by the ingest script |

### Example `.env` files

**Default local setup**

```env
OLLAMA_URL=http://localhost:11434
MODEL_NAME=iKhalid/ALLaM:7b
EMBED_MODEL=bge-m3:latest
CHROMA_PATH=./db/chroma
COLLECTION_NAME=finance_concepts
DOCS_PATH=./data/documents
NUM_CTX=4096
TEMPERATURE=0.3
MAX_HISTORY=10
PROMPTS_DIR=./prompts
```

**Custom paths** (e.g. documents on another drive)

```env
DOCS_PATH=D:/FinanceDocs
CHROMA_PATH=D:/X-Invest/db/chroma
```

For deeper architecture and module-level detail, see [TECHNICAL.md](TECHNICAL.md).

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Backend** | Python 3.12+, FastAPI, Uvicorn |
| **LLM & Embeddings** | ALLaM 7B + nomic-embed-text via Ollama |
| **Vector Database** | ChromaDB (persistent local storage) |
| **Retrieval** | Hybrid BM25 + semantic (rank-bm25) |
| **Market Data** | yfinance |
| **Machine Learning** | scikit-learn, XGBoost, pandas, numpy |
| **Frontend** | HTML5, Vanilla CSS/JS, Chart.js, Jinja2 |

No LangChain or paid LLM API keys are required for core functionality.

---

## Project Structure

```
X-Invest/
├── main.py                     # FastAPI entry point & page routing
├── config.py                   # Settings loaded from .env
├── requirements.txt
├── .env.example                # Configuration template
│
├── api/                        # HTTP endpoints
│   ├── chat.py                 # Streaming chat
│   ├── market.py               # Markets dashboard
│   ├── signal.py               # ML signal badge
│   └── backtest_api.py         # Backtest simulation
│
├── pipeline/                   # Chat context & response pipeline
├── services/llm_service.py     # Ollama chat + stream wrappers
│
├── rag/
│   ├── core/                   # Embeddings, ChromaDB, hybrid retriever
│   ├── preprocessing/          # Document ingest (python -m rag.preprocessing.ingest)
│   └── online/                 # Live market & news fetchers
│
├── market/                     # Dashboard feeds & curated tickers
├── prediction/                 # ML training, signals, backtest engine
│
├── templates/                  # Jinja2 HTML pages
├── static/                     # CSS, JS, images
├── prompts/                    # System prompt (system_prompt.txt)
├── db/                         # ChromaDB + caches (created at runtime)
└── data/documents/             # Your PDF/DOCX/TXT knowledge-base files
```

---

## Optional Setup Steps

### Ingest documents (RAG knowledge base)

1. Place files in `data/documents/` (or set `DOCS_PATH` in `.env`).
2. Run:
   ```bash
   python -m rag.preprocessing.ingest
   ```
3. Re-run after adding or replacing documents. The script recreates the Chroma collection for a clean slate.

### Train prediction models (Backtest & ML signals)

```bash
python prediction/train.py
```

Saves classifiers under `prediction/saved_models/`. Required for the Backtest page and `[Prediction]` context in chat.

---

## Command-Line Backtest

Run simulations without starting the web server:

```bash
python prediction/backtest.py
```

You will be prompted for tickers, date range, and starting capital. Metrics (Sharpe, drawdown, win rate) and an equity chart are printed/saved to disk.

---

## API Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/chat/stream` | Streaming chat (NDJSON) |
| `POST` | `/api/chat` | Blocking chat (JSON) |
| `POST` | `/api/clear` | Clear session memory |
| `GET` | `/api/market/dashboard` | Cached macro strip + company matrix |
| `GET` | `/api/market/{ticker}/history` | Price history for charts |
| `GET` | `/api/signal/{ticker}` | ML direction signal |
| `POST` | `/api/backtest` | Run strategy simulation |

Interactive docs: **http://localhost:8000/docs**

<details>
<summary><strong>Example: POST /api/backtest</strong></summary>

Request:

```json
{
  "ticker": "AAPL",
  "initial_capital": 10000.0,
  "start": "2024-01-01"
}
```

Response (abbreviated):

```json
{
  "success": true,
  "equity": [10000.0, 10050.2, 10210.5],
  "trades": [{ "entry": "...", "exit": "...", "pnl": 421.2 }],
  "metrics": {
    "total_return": 0.238555,
    "sharpe": 0.41,
    "win_rate": 0.6216,
    "max_drawdown": -0.29316
  }
}
```

</details>

---

## Troubleshooting

| Problem | What to try |
|---|---|
| **Connection refused to Ollama** | Confirm Ollama is running (`ollama list`). Check `OLLAMA_URL` in `.env`. |
| **Model not found** | `ollama pull <MODEL_NAME>` and `ollama pull <EMBED_MODEL>`. |
| **Chat works but no document answers** | Run ingest; verify files exist under `DOCS_PATH`. |
| **Backtest / signals unavailable** | Run `python prediction/train.py` and check `prediction/saved_models/`. |
| **Slow first page load on Markets** | Normal on cold start — startup pre-warm takes ~10s. Subsequent loads use cache. |
| **Unicode errors on Windows** | `main.py` sets UTF-8 on stdout; use PowerShell or Windows Terminal. |
| **Changed embedding model** | Re-run `python -m rag.preprocessing.ingest` so vectors match the new model. |

---

## Disclaimer

X-Invest is an educational project and does not offer professional investment advice. All AI responses include an automated disclaimer. Always consult a certified financial planner before making real investment decisions.

---

## License

MIT License. Free to use, modify, and build upon.
