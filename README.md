# X-Invest — AI Financial Assistant & Backtest Simulator

> An open-source, educational AI chatbot and trading strategy backtest simulator for stock market research.
> Answers finance questions in **Arabic and English**, retrieves **live market data**, explains financial concepts grounded in a document knowledge base, and simulates algorithmic trading strategies.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square)
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
- **RAG Pipeline** — semantically searches and retrieves context from your local document knowledge base (CFA guides, books, etc.) using ChromaDB.
- **Markets Dashboard** — browse 19 curated US and Middle Eastern companies (NASDAQ, NYSE, EGX, Tadawul) with an instant-load live data panel.
- **Preloading Caching Layer** — uses parallel macro downloads and launch-time pre-caching to guarantee instantaneous Markets dashboard page loads.
- **Backtest Simulator** — test strategy performance on historical data, rendering premium equity charts, drawdown calculations, and a complete trade ledger.
- **ML Signal Engine** — pluggable classification models (Random Forest) predicting 5-day forward return directions (Bullish, Bearish, or Neutral).

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Backend** | Python 3.10+, FastAPI, Uvicorn |
| **LLM & Embeddings** | ALLaM 7B / bge-m3:latest via Ollama (run locally) |
| **Vector Database** | ChromaDB (persistent local storage) |
| **Market Data** | yfinance API |
| **Machine Learning** | scikit-learn, pandas, numpy |
| **Frontend** | HTML5, Vanilla CSS, JavaScript, Chart.js, Jinja2 Templates |

No LangChain or external API keys are required. Runs fully offline on local hardware.

---

## Project Structure

```
X-Invest/
│
├── main.py                     # FastAPI entry point & page routing (starts cache warmup)
├── config.py                   # App configurations loaded from .env
├── requirements.txt            # System dependencies
├── setup_env.bat               # Windows automatic environment installer
├── .env.example                # Template configuration file
│
├── api/                        # HTTP API Endpoints
│   ├── chat.py                 # Chat and streaming endpoint handlers
│   ├── market.py               # Markets dashboard and history endpoints
│   ├── signal.py               # ML signal query endpoint
│   └── backtest_api.py         # Backtest simulation endpoint
│
├── market/                     # Live dashboard feeds
│   ├── companies.py            # Curated US, EGX, and Tadawul stocks list
│   ├── dashboard.py            # Single-company yfinance details fetcher
│   └── dashboard_feed.py       # Parallel macro feeds & dashboard caching layer
│
├── pipeline/                   # Context compilation & LLM response pipeline
│   ├── context_builder.py      # Combines RAG context + yfinance live data
│   ├── memory_manager.py       # Per-session conversation memory
│   ├── entity_extractor.py     # Extracts stock tickers from bilingual queries
│   ├── data_fetcher.py         # Formats stock data for LLM context ingestion
│   ├── online_rag.py           # In-memory session store for market records
│   └── postprocessor.py        # Appends financial disclaimers to LLM responses
│
├── rag/                        # Knowledge Base ingestion and retrieval
│   ├── ingest.py               # Parses, splits, embeds, and stores local PDFs/Docs
│   └── retriever.py            # MMR document search against ChromaDB
│
├── prediction/                 # ML prediction engine (Random Forest)
│   ├── train.py                # Pipeline feature builder and classifier trainer
│   ├── predict.py              # Evaluates models and outputs directional predictions
│   ├── signal_engine.py        # public signal getter API
│   ├── backtest.py             # CLI backtester and metrics compiler
│   ├── backtest_app.py         # Streamlit-based interactive backtest report
│   └── saved_models/           # trained models (.pkl)
│
├── templates/                  # Jinja2 HTML layout pages
│   ├── index.html              # Home page / landing dashboard
│   ├── chat.html               # Floating chat conversation interface
│   └── market.html             # Ticker panel + technical analysis matrix
│
├── static/                     # Frontend static assets
│   ├── css/style.css           # Custom theme stylesheet
│   └── js/                     # Frontend scripts (chat.js, market.js, backtest.js)
│
├── db/                         # Persistent databases (ChromaDB + Local Caches)
└── data/documents/             # Local PDF, DOCX, TXT documents folder
```

---

## Step-by-Step Installation Guide

Follow these steps to set up and run X-Invest on your local machine:

### 1. Prerequisites
Ensure you have the following installed on your system:
- **Git**
- **Python 3.10 or 3.11**
- **Conda** (Miniconda / Anaconda) installed and activated
- **Ollama** installed from [ollama.com](https://ollama.com)

### 2. Clone the Repository
Open your terminal or command prompt and clone the project:
```bash
git clone https://github.com/your-username/X-Invest.git
cd X-Invest
```

### 3. Initialize Python Environment
Choose **one** of the options below to set up your packages:

*   **Option A: Automatic Setup (Windows Only)**
    Double-click the `setup_env.bat` file in the project folder. This will automatically set up a virtual Conda environment named `xinvest` and install all required libraries.
    
*   **Option B: Manual Setup (Command Line)**
    Run the following commands in your terminal:
    ```bash
    # Create the conda environment
    conda create -n xinvest python=3.10 -y
    
    # Activate the environment
    conda activate xinvest
    
    # Install dependencies
    pip install -r requirements.txt
    ```

### 4. Download local AI Models
Start the Ollama application on your computer, then pull the LLM and embedding models in your command prompt:
```bash
# Pull the text embedding model
ollama pull bge-m3:latest

# Pull the Arabic-first chat model (ALLaM 7B)
ollama pull iKhalid/ALLaM:7b
```

### 5. Configure Environment Settings
Create your local `.env` configuration file:
```bash
cp .env.example .env
```
*(By default, the settings in `.env` are configured to match the local Ollama and ChromaDB paths, meaning it works out of the box).*

---

## Step-by-Step Execution Guide

Once installation is complete, follow these steps to prepare the data and start the application:

### Step 1: Ingest Documents into the Knowledge Base (Optional)
If you want the chatbot to answer questions based on your local files:
1. Drop your `.pdf`, `.docx`, `.txt`, or `.md` files into the `data/documents/` directory.
2. Run the ingest script to embed and save them to ChromaDB:
   ```bash
   python -m rag.ingest
   ```

### Step 2: Train the Prediction Models (Required for Backtesting)
The Backtest simulator page and model signals require trained classifier models to function. Run the training script once:
```bash
python prediction/train.py
```
This script downloads historical datasets, computes indicators, trains the Random Forest classifier models, and saves them under `prediction/saved_models/`.

### Step 3: Run the Web Application
Start the FastAPI server:
```bash
uvicorn main:app --reload
```
> [!NOTE]
> On server startup, the application runs a synchronous pre-caching sequence to fetch macro and ticker indicators from `yfinance`. This will block the terminal for ~10 seconds. Once completed, Uvicorn will display `Application startup complete` and run on `http://localhost:8000`.

Open `http://localhost:8000` in your web browser.

---

## Command Line Backtest (CLI)

You can also run simulations directly in the terminal without starting the web server.

### Run CLI Backtester
Run the backtest CLI script:
```bash
python prediction/backtest.py
```
The terminal will prompt you to input:
1. Stock symbols (comma-separated, e.g. `AAPL,MSFT,GOOGL`).
2. Start and end dates.
3. Starting capital.

The CLI downloads historical data, performs simulations, prints performance metrics (Sharpe, Drawdowns, Win Rate, Alpha), and saves an equity curve chart to disk (e.g. `AAPL_backtest_v14.png`).

---

## API Reference

### POST `/api/chat/stream`
Streaming chat endpoint used by the frontend.
```json
// Request Body
{ "session_id": "uuid", "message": "What is EPS?" }

// Response (Yielded NDJSON lines)
{"message": {"role": "assistant", "content": "Earnings"}, "done": false}
{"message": {"role": "assistant", "content": " Per Share"}, "done": false}
{"x_invest_final": true, "full_response": "...", "session_id": "uuid"}
```

### POST `/api/backtest`
Simulates a backtest for a ticker symbol over a given date range.
```json
// Request Body
{
  "ticker": "AAPL",
  "initial_capital": 10000.0,
  "start": "2024-01-01"
}

// Response Body
{
  "success": true,
  "equity": [10000.0, 10050.2, 10210.5],
  "trades": [
    {
      "entry": "2024-01-15 00:00:00",
      "exit": "2024-01-22 00:00:00",
      "entry_price": 182.5,
      "exit_price": 190.2,
      "shares": 54.7,
      "pnl": 421.2,
      "pnl_pct": 0.0422,
      "hold_days": 7,
      "reason": "take_profit"
    }
  ],
  "metrics": {
    "ticker": "AAPL",
    "final_capital": 12385.55,
    "total_trades": 37,
    "total_return": 0.238555,
    "ann_return": 0.113313,
    "max_drawdown": -0.29316,
    "max_dd": -0.29316,
    "sharpe": 0.41,
    "win_rate": 0.6216,
    "profit_factor": 1.339
  }
}
```

### GET `/api/market/dashboard`
Returns the cached macro strip values and the company metrics matrix instantaneously.

---

## Disclaimer

X-Invest is an educational project and does not offer professional investment advice. All AI responses include an automated disclaimer. Always consult a certified financial planner before making real investment decisions.

---

## License
MIT License. Free to use, modify, and build upon.
