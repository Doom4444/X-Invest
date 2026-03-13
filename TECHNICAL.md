# X-Invest — Technical Documentation
**Version:** MVP Phase 1  
**Project:** BIS Graduation Project — Egypt, July 2026  
**Stack:** Python · FastAPI · Ollama · ChromaDB · yfinance  

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Architecture at a Glance](#2-architecture-at-a-glance)
3. [Layer 0 — Startup: `main.py` + `config.py`](#3-layer-0--startup-mainpy--configpy)
4. [Layer 1 — The Browser Gets a Page](#4-layer-1--the-browser-gets-a-page)
5. [Layer 2 — User Sends a Message: `chat.js` → `api/chat.py`](#5-layer-2--user-sends-a-message-chatjs--apichatpy)
6. [Layer 3 — The Server Receives It: `api/chat.py`](#6-layer-3--the-server-receives-it-apichatpy)
7. [Layer 4 — Context Assembly: `pipeline/context_builder.py`](#7-layer-4--context-assembly-pipelinecontext_builderpy)
8. [Layer 5 — Memory: `pipeline/memory_manager.py`](#8-layer-5--memory-pipelinememory_managerpy)
9. [Layer 6 — The LLM Call: `services/llm_service.py`](#9-layer-6--the-llm-call-servicesllm_servicepy)
10. [Layer 7 — Streaming Back to the Browser](#10-layer-7--streaming-back-to-the-browser)
11. [Layer 8 — Postprocessor: `pipeline/postprocessor.py`](#11-layer-8--postprocessor-pipelinepostprocessorpy)
12. [Layer 9 — Browser Renders the Stream: `chat.js`](#12-layer-9--browser-renders-the-stream-chatjs)
13. [The Market Page — Independent System](#13-the-market-page--independent-system)
14. [The RAG Ingest — Offline Process: `rag/ingest.py`](#14-the-rag-ingest--offline-process-ragingestpy)
15. [Complete Request Trace — One Full Message End to End](#15-complete-request-trace--one-full-message-end-to-end)
16. [Module Reference](#16-module-reference)

---

## 1. Project Overview

X-Invest is a vertical AI financial assistant — meaning it is intentionally scoped to finance only and refuses all other topics. The system is built entirely from scratch without frameworks like LangChain. Every component is custom-written so the architecture is fully transparent and debuggable.

**What makes it "vertical":**
- The system prompt in `prompts/system_prompt.txt` explicitly instructs the model to refuse non-finance questions
- The postprocessor enforces a disclaimer on every single response
- The RAG knowledge base contains only finance documents

**What makes it "grounded":**
- Every response is built from three information sources: a static knowledge base, live market data, and session memory
- The model is instructed never to invent numbers — if data is unavailable it must say so

**What makes it "local":**
- The LLM (ALLaM 7B) runs on your machine via Ollama — no OpenAI API, no external calls
- Embeddings also run locally via Ollama's nomic-embed-text model
- ChromaDB stores vectors on disk at `db/chroma/`
- yfinance fetches live data directly from Yahoo Finance

---

## 2. Architecture at a Glance

```
Browser (HTML/CSS/JS)
    │
    │  GET /  GET /chat  GET /market    ← page loads
    │  POST /api/chat/stream            ← chat messages
    │  GET  /api/market/companies       ← company list
    │  GET  /api/market/{ticker}        ← dashboard data
    │
FastAPI  (main.py)
    │
    ├── api/chat.py          ← chat endpoints
    │     │
    │     ├── pipeline/context_builder.py
    │     │     ├── rag/retriever.py          ← ChromaDB + MMR
    │     │     ├── pipeline/entity_extractor.py
    │     │     ├── pipeline/data_fetcher.py  ← yfinance
    │     │     └── pipeline/online_rag.py    ← session store
    │     │
    │     ├── pipeline/memory_manager.py      ← conversation history
    │     ├── services/llm_service.py         ← Ollama API
    │     └── pipeline/postprocessor.py       ← disclaimer
    │
    ├── api/market.py        ← market endpoints
    │     ├── market/companies.py
    │     └── market/dashboard.py             ← yfinance
    │
    └── api/signal.py        ← prediction endpoints (stub)
          └── prediction/signal_engine.py
```

---

## 3. Layer 0 — Startup: `main.py` + `config.py`

When you run `uvicorn main:app --reload`, two things happen before any request arrives.

### `config.py` runs first

```python
from dotenv import load_dotenv
import os
load_dotenv()   # reads .env file, populates environment variables

OLLAMA_URL    = os.getenv("OLLAMA_URL",    "http://localhost:11434")
MODEL_NAME    = os.getenv("MODEL_NAME",    "iKhalid/ALLaM:7b")
EMBED_MODEL   = os.getenv("EMBED_MODEL",   "nomic-embed-text:latest")
NUM_CTX       = int(os.getenv("NUM_CTX",   "4096"))
TEMPERATURE   = float(os.getenv("TEMPERATURE", "0.3"))
MAX_HISTORY   = int(os.getenv("MAX_HISTORY",   "10"))
CHROMA_PATH   = os.getenv("CHROMA_PATH",   "./db/chroma")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "finance_concepts")
PROMPTS_DIR   = os.getenv("PROMPTS_DIR",   "./prompts")
DOCS_PATH     = os.getenv("DOCS_PATH",     "./data/documents")
```

Every other file imports from here using `from config import MODEL_NAME`. No file ever calls `os.getenv()` directly. This single-source-of-truth pattern means changing the model name requires editing one line in `.env`, not hunting through 6 files.

### `main.py` runs second

```python
app = FastAPI(title="X-Invest", version="1.0.0")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
```

This creates the FastAPI application, mounts the `/static` folder (so the browser can request CSS/JS/images directly), and sets up Jinja2 to render HTML templates.

Then it calls `_include_routers()`:

```python
def _include_routers() -> None:
    for module_name in ("api.chat", "api.market", "api.signal"):
        try:
            mod = __import__(module_name, fromlist=["router"])
            app.include_router(mod.router)
        except Exception as e:
            logger.exception("Failed to include router %s: %s", module_name, e)
```

This imports each `api/` file and registers its URL routes with FastAPI. The `try/except` means if one router has an import error, the other two still load — the app does not crash completely.

Finally, `main.py` registers the three page routes:

```python
@app.get("/")
async def home(request: Request):
    ctx = {"request": request, **_home_bg_context()}
    return templates.TemplateResponse("index.html", ctx)

@app.get("/chat")
async def chat_page(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})

@app.get("/market")
async def market_page(request: Request):
    return templates.TemplateResponse("market.html", {"request": request})
```

These routes contain no business logic. They just give the browser the HTML file.

---

## 4. Layer 1 — The Browser Gets a Page

When you visit `http://localhost:8000/chat`, FastAPI returns `templates/chat.html`. The browser then automatically makes two more requests:

- `GET /static/css/style.css` — the stylesheet (served by the StaticFiles mount)
- `GET /static/js/chat.js` — the JavaScript

Once `chat.js` is loaded, it runs immediately. The first thing it does is establish a session identity:

```javascript
const sessionId =
  window.localStorage.getItem("xinvest_session") ||
  (window.crypto.randomUUID && window.crypto.randomUUID()) ||
  String(Date.now());
window.localStorage.setItem("xinvest_session", sessionId);
```

This UUID is the key that ties your entire conversation together on the server side. Every request you send includes it. If you refresh the page, `localStorage.getItem` returns the same UUID — that is why the server remembers your conversation across refreshes (as long as the server has not restarted, since memory is in-memory only on the server).

---

## 5. Layer 2 — User Sends a Message: `chat.js` → `api/chat.py`

When you type a message and press Enter or click Send, `sendMessage()` runs.

### Step 1 — Language detection

Before sending anything, the JS scans your text for Unicode character ranges:

```javascript
function detectLanguageHint(text) {
  if (/[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]/.test(text))
    return "The user wrote in Arabic. You MUST respond entirely in Arabic.";
  return "";
}
```

`\u0600-\u06FF` is the Unicode block for Arabic script. If your message contains Arabic characters, a hidden instruction gets prepended to the message that is actually sent to the backend:

```
[The user wrote in Arabic. You MUST respond entirely in Arabic.]
ما هو سعر سهم أبل الآن؟
```

The user's bubble in the UI shows only their original text. The backend receives the augmented version. This reinforces the system prompt's language rule, because LLMs sometimes "drift" to their dominant training language under certain conditions.

### Step 2 — UI updates immediately (before the fetch)

Two DOM elements are created before any network request:

```javascript
createMessageRow("user", text);           // user bubble with their text
const { bubble, meta } = createMessageRow("assistant", "");  // empty assistant bubble
setBubble(bubble, "", true);              // true = show blinking cursor
```

The user sees their message appear instantly and an empty assistant bubble with a cursor. This is important UX — it communicates "the system received your message and is working."

### Step 3 — The fetch

```javascript
const res = await fetch("/api/chat/stream", {
  method:  "POST",
  headers: { "Content-Type": "application/json" },
  body:    JSON.stringify({ session_id: sessionId, message: messageToSend }),
});
```

The request body contains the session ID and the augmented message. The response is a streaming `ReadableStream` — tokens arrive continuously rather than all at once.

---

## 6. Layer 3 — The Server Receives It: `api/chat.py`

`chat_stream_endpoint` is the orchestrator. It calls everything else in the correct order.

```python
body       = await request.json()
session_id = body.get("session_id", "default")
user_query = body.get("message", "").strip()
```

### Critical ordering — context BEFORE history

```python
context = build_context(user_query, session_id=session_id)  # ← Step 1
add_message(session_id, "user", user_query)                  # ← Step 2
history = get_history(session_id)                            # ← Step 3
```

**Why this order is important:**  
`build_context` does semantic retrieval using only the raw query. If you added the message to history first, the query vector used for RAG search would contain the noise of previous conversation turns. A pure single-message query produces cleaner, more relevant retrieval results.

### The streaming generator

```python
full_response_tokens: list[str] = []

def generate():
    for chunk in stream_chat(history, context=context):
        yield chunk    # → sent to browser immediately

        # Also parse to track state server-side
        for line in chunk.decode("utf-8").splitlines():
            data = json.loads(line)
            token = data.get("message", {}).get("content", "")
            if token:
                full_response_tokens.append(token)

            if data.get("done"):
                full_text  = "".join(full_response_tokens)
                final_text = process(full_text)               # postprocessor
                add_message(session_id, "assistant", final_text)
                trim(session_id)
                sentinel = json.dumps({
                    "x_invest_final": True,
                    "full_response":  final_text,
                    "session_id":     session_id,
                }) + "\n"
                yield sentinel.encode("utf-8")

return StreamingResponse(generate(), media_type="application/x-ndjson")
```

The generator does two things simultaneously:
1. It yields each raw chunk from Ollama straight to the browser (low latency — tokens appear as fast as the model generates them)
2. It also parses those same chunks to accumulate the full response, run postprocessing, save to memory, and send a final "sentinel" line

The sentinel is a custom NDJSON line with `x_invest_final: true`. It carries the postprocessed, disclaimer-guaranteed version of the full response so the browser can replace the streamed text with the clean final version.

---

## 7. Layer 4 — Context Assembly: `pipeline/context_builder.py`

`build_context(query, session_id)` is the most architecturally important function in the project. It assembles the information block that gets injected into the model's input. It has three sources.

### Source 1 — Static Knowledge Base (RAG) via `rag/retriever.py`

```python
chunks = retrieve(query, n_results=3)
if chunks:
    kb_text = "\n---\n".join(c["text"] for c in chunks)
    sections.append(f"[Knowledge Base]:\n{kb_text}")
```

Inside `rag/retriever.py`, `retrieve()` does the following:

**Step A — Embed the query**

```python
r = requests.post(
    f"{OLLAMA_URL}/api/embeddings",
    json={"model": EMBED_MODEL, "prompt": text},
    timeout=30
)
return r.json()["embedding"]   # a list of 768 floats
```

This sends your query text to Ollama's embedding endpoint. `nomic-embed-text` converts the text into a 768-dimensional vector — a list of 768 numbers that represent the meaning of the text mathematically. Semantically similar texts produce vectors that are close together in this 768-dimensional space.

**Step B — Query ChromaDB for candidates**

```python
pool_size = min(20, col.count())
res = col.query(
    query_embeddings=[emb],
    n_results=pool_size,
    include=["documents", "metadatas", "distances", "embeddings"]
)
```

ChromaDB finds the 20 most similar stored chunks by computing cosine distance between the query vector and every stored chunk's vector. This is the "candidate pool" for MMR to work from.

**Step C — MMR reranking**

Plain top-K retrieval has a problem: the top 3 results are often nearly identical. If your document has a paragraph about P/E ratios mentioned five times, you'd get three chunks all saying the same thing.

MMR (Maximal Marginal Relevance) solves this by balancing relevance with diversity:

```python
def mmr_score(i):
    relevance  = cosine(emb, embeddings[i])
    redundancy = max(cosine(embeddings[i], embeddings[s]) for s in selected)
    return lambda_ * relevance - (1 - lambda_) * redundancy
```

With `lambda_ = 0.5`:
- **Relevance** = how similar is this chunk to the query?
- **Redundancy** = how similar is this chunk to chunks already selected?
- **MMR score** = 50% relevance − 50% redundancy

The algorithm greedily picks the candidate with the highest MMR score, adds it to the selected set, and repeats. The result is 3 chunks that are all relevant but cover different angles of the topic.

A distance threshold of `1.2` filters out chunks that are not relevant enough even if they were in the top pool.

### Source 2 — Live Market Data via `pipeline/entity_extractor.py` + `pipeline/data_fetcher.py`

```python
tickers = extract_tickers(query)
for ticker in tickers:
    data = get_stock_data(ticker)
    fmt  = format_for_prompt(data)
    live_parts.append(fmt)
    add_to_session(session_id, fmt)   # store for follow-up questions
```

`extract_tickers()` uses two detection methods:

```python
# Method 1 — Regex: matches explicit uppercase tickers
tickers.update(re.findall(r"\b[A-Z]{2,5}(?:\.[A-Z]{1,2})?\b", query))
# Matches: AAPL, TSLA, COMI.CA, 2222.SR etc.
# Min 2 chars to avoid matching "E" from "P/E ratio"
# Optional .XX suffix for EGX (.CA) and Saudi (.SR) markets

# Method 2 — Name dictionary lookup
for name, ticker in KNOWN_TICKERS.items():
    if name in query_lower or name in query:
        tickers.add(ticker)
# Maps: "apple" / "أبل" → "AAPL", "aramco" / "أرامكو" → "2222.SR" etc.
```

`get_stock_data()` calls `yf.Ticker(ticker).info` — a single yfinance call that returns a large dict from Yahoo Finance. It extracts the fields relevant for the LLM:

```python
return {
    "ticker":      ticker,
    "name":        info.get("longName", ticker),
    "price":       info.get("currentPrice") or info.get("regularMarketPrice"),
    "currency":    info.get("currency", "USD"),
    "pe_ratio":    info.get("trailingPE"),
    "market_cap":  info.get("marketCap"),
    "week52_high": info.get("fiftyTwoWeekHigh"),
    "week52_low":  info.get("fiftyTwoWeekLow"),
    "change_pct":  info.get("regularMarketChangePercent"),
    "news":        [n["title"] for n in (stock.news or [])[:3]],
}
```

`format_for_prompt()` converts this dict into readable text that the LLM can parse:

```python
lines = [
    f"Stock: {d['name']} ({d['ticker']})",
    f"Current Price: {d['price']} {d['currency']}",
    f"P/E Ratio: {d['pe_ratio']:.2f}",
    f"Today Change: {d['change_pct']:.2f}%",
    f"52W High: {d['week52_high']}",
    ...
]
```

The function also calls `add_to_session(session_id, fmt)` immediately after fetching, storing the formatted data in `online_rag.py` for follow-up questions.

### Source 3 — Session Store via `pipeline/online_rag.py`

```python
session_ctx = get_session_context(session_id)
if session_ctx and not any("[Live Market Data]" in s for s in sections):
    sections.append(session_ctx)
```

`online_rag.py` is a simple in-memory dict:

```python
_session_data: dict[str, list[str]] = defaultdict(list)
```

When you ask "what was Apple's price?" as a follow-up with no ticker in the message — entity extraction finds nothing and yfinance fetches nothing. But the session store still has the Apple data from your first message. The guard `not any("[Live Market Data]" in s ...)` prevents double-injecting if new live data was already found for this turn.

This is called "online RAG" — volatile, session-scoped context storage as opposed to the persistent static ChromaDB knowledge base.

### What the model actually sees

After all three sources are assembled:

```
[Knowledge Base]:
P/E Ratio is a valuation metric that measures how much investors pay
per dollar of earnings. A high P/E suggests growth expectations...
---
Earnings Per Share (EPS) represents the company's profit divided by
the number of outstanding shares...

[Live Market Data]:
Stock: Apple Inc (AAPL)
Current Price: 189.30 USD
P/E Ratio: 28.54
Today Change: -0.82%
52W High: 199.62
52W Low: 164.08
Latest News: Apple reports record Q4 earnings; iPhone 16 demand strong
```

This entire block is prepended to the user's message before the LLM sees it.

---

## 8. Layer 5 — Memory: `pipeline/memory_manager.py`

Conversation memory is stored in a plain Python dict at module level:

```python
_sessions: dict = defaultdict(list)
```

Each session ID maps to a list of message dicts:

```python
_sessions["uuid-abc"] = [
    {"role": "user",      "content": "What is P/E?"},
    {"role": "assistant", "content": "P/E ratio is a valuation metric..."},
    {"role": "user",      "content": "Give me an example"},
]
```

This exact format is what Ollama's `/api/chat` endpoint expects. The model reads the full history on every call — that is how it "remembers" previous messages. It is not magic; the entire conversation is literally resent on every request.

The trim function prevents unbounded memory growth:

```python
def _trim(session_id: str) -> None:
    h = _sessions[session_id]
    if len(h) > MAX_HISTORY * 2:          # MAX_HISTORY=10, so threshold is 20 messages
        _sessions[session_id] = h[-(MAX_HISTORY * 2):]   # keep only the last 20
```

With `MAX_HISTORY=10`, the model always sees the last 10 complete exchanges (user + assistant = 2 messages each = 20 total). Older context is dropped.

**Important limitation:** Memory is in-memory only. If the server restarts, all conversation history is lost. The `future/db.py` scaffold is designed for upgrading this to persistent database storage without changing the API.

---

## 9. Layer 6 — The LLM Call: `services/llm_service.py`

`stream_chat(history, context)` is called with two arguments:
- `history` — the full conversation list from memory_manager
- `context` — the assembled KB + live data block from context_builder

First, it enriches the messages with context:

```python
def _enrich(messages: list, context: str) -> list:
    enriched = messages.copy()           # never modify the original
    if context and enriched:
        enriched[-1]["content"] = context + "\n\n" + enriched[-1]["content"]
    return enriched
```

This prepends the context block to the last user message (index -1). The model sees:

```
[context block]

ما هو سعر سهم أبل الآن؟
```

The system prompt instructs the model to use the provided data rather than inventing numbers. The context block gives it the actual data to use.

Then it calls Ollama with streaming enabled:

```python
r = requests.post(
    f"{OLLAMA_URL}/api/chat",
    json={
        "model":    MODEL_NAME,
        "system":   SYSTEM_PROMPT,       # loaded from prompts/system_prompt.txt at startup
        "messages": _enrich(messages, context),
        "stream":   True,
        "options":  {"temperature": TEMPERATURE, "num_ctx": NUM_CTX}
    },
    stream=True,
    timeout=120
)
for chunk in r.iter_content(chunk_size=None):
    yield chunk    # raw bytes, passed through as-is
```

Key parameters:
- `temperature: 0.3` — low randomness. For a factual finance assistant, deterministic answers are more appropriate than creative ones
- `num_ctx: 4096` — the context window. The maximum number of tokens the model can hold in its "working memory" at once. The system prompt + context block + history + new message must all fit within this limit
- `stream: True` — Ollama sends one JSON object per line as it generates each token, rather than waiting for the full response

---

## 10. Layer 7 — Streaming Back to the Browser

The NDJSON stream from Ollama looks like this:

```json
{"message": {"role": "assistant", "content": "سهم"}, "done": false}
{"message": {"role": "assistant", "content": " أبل"}, "done": false}
{"message": {"role": "assistant", "content": " يتداول"}, "done": false}
...
{"done": true, "eval_count": 312, "eval_duration": 4200000000}
```

Each line is a complete JSON object. The content field contains one or a few tokens. The final line has `done: true` and includes performance stats.

Back in `api/chat.py`, the `generate()` function simultaneously:
1. Yields each raw chunk to the browser immediately (no buffering — minimum latency)
2. Parses those same chunks to track the full response server-side

After the `done: true` line, it appends the sentinel:

```json
{"x_invest_final": true, "full_response": "سهم أبل يتداول...\n\n---\nDisclaimer: ...", "session_id": "uuid"}
```

The `StreamingResponse` is returned with headers that disable caching and proxy buffering:

```python
return StreamingResponse(
    generate(),
    media_type="application/x-ndjson",
    headers={
        "Cache-Control":     "no-cache",
        "X-Accel-Buffering": "no",     # prevents nginx from buffering the stream
    }
)
```

---

## 11. Layer 8 — Postprocessor: `pipeline/postprocessor.py`

The simplest file in the project but an important safety net:

```python
DISCLAIMER = (
    "\n\n---\n"
    "Disclaimer: This information is for educational purposes only "
    "and is not professional financial or investment advice. "
    "Please consult a licensed financial advisor before making investment decisions."
)

def process(response: str) -> str:
    response = response.strip()
    keywords = ["disclaimer", "not professional", "educational purposes",
                "financial advisor", "not financial advice",
                "not investment advice"]
    if not any(kw in response.lower() for kw in keywords):
        response += DISCLAIMER
    return response
```

The system prompt instructs the model to add a disclaimer to every response. But models occasionally forget, especially on short factual answers. This code runs unconditionally after every response. It checks whether any disclaimer-related keyword appears in the response text, and if not, appends the full disclaimer. This guarantees the legal protection text is always present regardless of model behavior.

---

## 12. Layer 9 — Browser Renders the Stream: `chat.js`

The browser reads the NDJSON stream line by line:

```javascript
const reader  = res.body.getReader();
const decoder = new TextDecoder();
let   buffer  = "";

while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });

    // Split on newlines — each line is one JSON object
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";    // keep incomplete last fragment for next chunk

    for (const line of lines) {
        if (!line.trim()) continue;
        const data = JSON.parse(line);

        // Backend error (e.g. Ollama unreachable)
        if (data.error) {
            setBubble(bubble, "Sorry, something went wrong.\n" + data.error, false);
            return;
        }

        // Sentinel — stream fully done, replace with postprocessed version
        if (data.x_invest_final) {
            setBubble(bubble, data.full_response || fullText, false);
            continue;
        }

        // Normal token — append to bubble with cursor
        const token = data.message?.content ?? "";
        if (token) {
            fullText += token;
            setBubble(bubble, fullText, true);    // true = show blinking cursor
        }

        // Ollama done flag — show token/time stats
        if (data.done) {
            setBubble(bubble, fullText, false);   // false = hide cursor
            const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
            meta.textContent = `Tokens: ${data.eval_count} · Time: ${elapsed}s`;
        }
    }
}
```

The buffer trick is essential. A single network chunk from the browser's `ReadableStream` may contain multiple complete JSON lines, or it may cut a JSON object in half at a packet boundary. `lines.pop()` removes and saves the last (potentially incomplete) fragment, which gets prepended to the next incoming chunk. This ensures no JSON object is ever lost or malformed.

`setBubble()` rewrites the entire bubble's `innerHTML` on every token:

```javascript
function setBubble(bubble, text, showCursor = false) {
    bubble.innerHTML =
        escapeHtml(text) +
        (showCursor ? '<span class="cursor"></span>' : "");
    scrollToBottom();
}
```

`escapeHtml()` first converts the text to safe HTML (preventing XSS attacks), then appends the cursor span if still streaming. The CSS `.cursor` class animates the blink with a keyframe animation.

---

## 13. The Market Page — Independent System

The market page has zero dependency on the chat pipeline. It does not use ChromaDB, does not call the LLM, and works even if Ollama is down.

**Flow when you visit `/market`:**

1. Browser loads `market.html` and then `market.js`
2. `market.js` calls `GET /api/market/companies`
3. `api/market.py` returns the `COMPANIES` list from `market/companies.py` — a hardcoded list of 19 tickers (9 US, 5 EGX, 5 Saudi)
4. `market.js` renders a card for each company

**Flow when you click a company card:**

1. `market.js` calls `GET /api/market/AAPL` (or whichever ticker)
2. `api/market.py` calls `market/dashboard.py` → `get_dashboard_data("AAPL")`
3. `get_dashboard_data` calls `yf.Ticker("AAPL").info` — a richer yfinance call than the chat pipeline uses
4. Returns structured JSON with all display fields:
   ```python
   {
       "price":      189.30,
       "change":     -1.54,
       "change_pct": -0.81,
       "pe_ratio":   28.54,
       "pb_ratio":   45.2,
       "eps":        6.43,
       "dividend":   0.0053,
       "week52_high": 199.62,
       "week52_low":  164.08,
       "day_high":   190.22,
       "day_low":    188.54,
       "volume":     55234100,
       "news": [{"title": "...", "link": "..."}, ...],
   }
   ```
5. `market.js` renders the dashboard panel with this data

**Why two separate fetchers (`data_fetcher.py` and `dashboard.py`)?**

They serve different consumers:
- `data_fetcher.py` → produces compact Arabic text strings optimized for LLM context injection, minimal fields
- `dashboard.py` → produces full structured JSON optimized for UI rendering, all fields including links

Same data source (yfinance), different output format, different purpose.

---

## 14. The RAG Ingest — Offline Process: `rag/ingest.py`

`rag/ingest.py` runs as a standalone script, not as part of the web server:

```bash
python -m rag.ingest
```

It is run once (or re-run when new documents are added) to populate ChromaDB.

### Step 1 — Read documents

Supports four file types:
- **PDF** via `pdfplumber` — handles multi-column layouts and tables better than `pypdf`
- **DOCX** via `python-docx` — extracts paragraph text
- **TXT** — direct UTF-8 read with latin-1 fallback
- **MD** — read and strip markdown symbols (headings `##`, bold `**`, code blocks, links) so embeddings focus on semantic content

### Step 2 — Chunk text

```python
CHUNK_SIZE    = 400   # words per chunk
CHUNK_OVERLAP = 50    # words shared between consecutive chunks

while start < len(words):
    end        = min(start + CHUNK_SIZE, len(words))
    chunk_text = " ".join(words[start:end])
    chunks.append({"text": chunk_text, "source": source, "chunk_index": idx})
    start = end - CHUNK_OVERLAP    # step back 50 words before next chunk
```

The 50-word overlap is critical. Imagine a financial concept definition that spans two paragraphs. Without overlap, it gets split in half and neither chunk contains the complete concept. With overlap, the boundary content appears in both adjacent chunks, making the full concept retrievable.

400 words ≈ 600–800 tokens, which fits comfortably inside `nomic-embed-text`'s 8192-token context window.

### Step 3 — Embed chunks

Each chunk is sent to Ollama:

```python
r = requests.post(
    f"{OLLAMA_URL}/api/embeddings",
    json={"model": EMBED_MODEL, "prompt": chunk["text"]},
    timeout=30
)
embedding = r.json()["embedding"]   # 768-dim vector
```

### Step 4 — Store in ChromaDB

```python
client = chromadb.PersistentClient(path=CHROMA_PATH)
client.delete_collection(COLLECTION_NAME)   # rebuild from scratch
col = client.create_collection(COLLECTION_NAME)
col.add(documents=docs, embeddings=embeddings, ids=ids, metadatas=metas)
```

ChromaDB stores the text and its vector together. When `retrieve()` queries ChromaDB later, it compares the query's vector against all stored vectors and returns the most similar chunks by cosine distance.

The collection is **rebuilt from scratch** every time ingest runs. This means adding new documents is safe: just drop them in and re-run. No merging logic required.

---

## 15. Complete Request Trace — One Full Message End to End

```
User types: "ما هو سعر سهم أبل الآن؟"
                │
    ── chat.js ──────────────────────────────────────────────────────────
                │
    detectLanguageHint() → Arabic detected
    messageToSend = "[The user wrote in Arabic. You MUST respond entirely in Arabic.]\nما هو سعر سهم أبل الآن؟"
                │
    createMessageRow("user", "ما هو سعر سهم أبل الآن؟")   ← user bubble appears
    createMessageRow("assistant", "")                      ← empty bubble + cursor appears
                │
    fetch("POST /api/chat/stream", { session_id, message: messageToSend })
                │
    ── api/chat.py ──────────────────────────────────────────────────────
                │
    context = build_context(user_query, session_id)
                │
    ── pipeline/context_builder.py ──────────────────────────────────────
                │
        ── rag/retriever.py ─────────────────────────────────────────────
                │
            embed("ما هو سعر سهم أبل الآن؟")
            → POST /api/embeddings to Ollama
            → returns [0.023, -0.441, 0.187, ...] (768 floats)
                │
            ChromaDB.query(query_embeddings=[...], n_results=20)
            → returns 20 candidate chunks with distances + embeddings
                │
            MMR selects 3 diverse, relevant chunks
            → [{"text": "Apple Inc is a technology company...", "distance": 0.34},
               {"text": "P/E Ratio measures how much investors pay...", "distance": 0.41},
               {"text": "Stock price is determined by supply and demand...", "distance": 0.52}]
                │
        ── pipeline/entity_extractor.py ─────────────────────────────────
                │
            regex finds: []   (no uppercase tickers in Arabic text)
            dict lookup: "أبل" → "AAPL"
            returns: ["AAPL"]
                │
        ── pipeline/data_fetcher.py ─────────────────────────────────────
                │
            yf.Ticker("AAPL").info
            → { currentPrice: 189.30, trailingPE: 28.54, ... }
                │
            format_for_prompt() →
            "Stock: Apple Inc (AAPL)\nCurrent Price: 189.30 USD\nP/E Ratio: 28.54\n..."
                │
        ── pipeline/online_rag.py ───────────────────────────────────────
                │
            add_to_session("uuid-abc", "Stock: Apple Inc...")
            (stored for follow-up questions in this session)
                │
    context = """
    [Knowledge Base]:
    Apple Inc is a technology company...
    ---
    P/E Ratio measures how much investors pay...
    ---
    Stock price is determined by supply and demand...

    [Live Market Data]:
    Stock: Apple Inc (AAPL)
    Current Price: 189.30 USD
    P/E Ratio: 28.54
    Today Change: -0.82%
    52W High: 199.62
    52W Low: 164.08
    """
                │
    add_message("uuid-abc", "user", user_query)
    history = [{"role": "user", "content": "[Arabic hint]\nما هو سعر سهم أبل الآن؟"}]
                │
    ── services/llm_service.py ──────────────────────────────────────────
                │
    _enrich(history, context):
    history[-1]["content"] = context + "\n\n" + history[-1]["content"]
                │
    POST http://localhost:11434/api/chat
    {
        model: "iKhalid/ALLaM:7b",
        system: "You are X-Invest, a specialized educational financial assistant...",
        messages: [{ role: "user", content: "[context block]\n\n[Arabic hint]\nما هو سعر سهم أبل الآن؟" }],
        stream: true,
        options: { temperature: 0.3, num_ctx: 4096 }
    }
                │
    Ollama generates and streams NDJSON:
    {"message": {"role": "assistant", "content": "سهم"}, "done": false}
    {"message": {"role": "assistant", "content": " أبل"}, "done": false}
    {"message": {"role": "assistant", "content": " (AAPL)"}, "done": false}
    ...
    {"done": true, "eval_count": 156, "eval_duration": 3800000000}
                │
    ── api/chat.py generate() ───────────────────────────────────────────
                │
    each chunk → yield to browser immediately
    each chunk → parse, accumulate full_response_tokens
                │
    on done:
        full_text  = "سهم أبل (AAPL) يتداول حالياً عند سعر 189.30 دولار..."
        final_text = process(full_text)   ← postprocessor checks/adds disclaimer
        add_message("uuid-abc", "assistant", final_text)
        trim("uuid-abc")
        yield {"x_invest_final": true, "full_response": final_text}
                │
    ── chat.js stream reader ────────────────────────────────────────────
                │
    each token chunk:
        fullText += token
        setBubble(bubble, fullText, true)    ← bubble updates, cursor visible
                │
    on done:
        meta.textContent = "Tokens: 156 · Time: 4.2s"
                │
    on x_invest_final:
        setBubble(bubble, data.full_response, false)   ← final version, cursor gone
                │
    User sees: Full Arabic response with live AAPL price + disclaimer
```

---

## 16. Module Reference

| File | Purpose | Called By |
|---|---|---|
| `main.py` | FastAPI app, router registration, page routes | Entry point |
| `config.py` | All settings from `.env` | All modules |
| `api/chat.py` | `/api/chat` and `/api/chat/stream` endpoints | `chat.js` |
| `api/market.py` | `/api/market/companies` and `/api/market/{ticker}` | `market.js` |
| `api/signal.py` | `/api/signal/{ticker}` | `market.js` |
| `pipeline/context_builder.py` | Orchestrates all retrieval, assembles context block | `api/chat.py` |
| `pipeline/memory_manager.py` | Per-session conversation history | `api/chat.py` |
| `pipeline/entity_extractor.py` | Detects stock tickers in Arabic and English text | `context_builder.py` |
| `pipeline/data_fetcher.py` | Fetches live yfinance data, formats for LLM | `context_builder.py` |
| `pipeline/online_rag.py` | Volatile per-session store for fetched live data | `context_builder.py`, `api/chat.py` |
| `pipeline/postprocessor.py` | Guarantees disclaimer is present in every response | `api/chat.py` |
| `rag/retriever.py` | MMR semantic search against ChromaDB | `context_builder.py` |
| `rag/ingest.py` | Offline: reads docs, chunks, embeds, stores in ChromaDB | Run manually |
| `services/llm_service.py` | Ollama blocking + streaming chat client | `api/chat.py` |
| `market/companies.py` | Curated list of 19 tickers | `api/market.py` |
| `market/dashboard.py` | Fetches full yfinance data for Market UI panel | `api/market.py` |
| `prediction/signal_engine.py` | Public interface for prediction module (stub) | `api/signal.py` |
| `models/schemas.py` | Pydantic request/response validation models | `api/` files |
| `prompts/system_prompt.txt` | ALLaM system prompt — defines bot persona and rules | `services/llm_service.py` |
| `static/js/chat.js` | Streaming chat UI, session management, language detection | Browser |
| `static/js/market.js` | Market page — company list and dashboard rendering | Browser |
| `static/css/style.css` | All styles for all 3 pages | Browser |

---

*Documentation covers MVP Phase 1. Prediction module (prediction/signal_engine.py) is stubbed and covered under a separate implementation document.*
