# X-Invest — Contributor Guide

> For the Prediction Team (A, B, C), the RAG Teammate, and reference for Adham.
> Read this once before you touch any file. It covers everything you need.

---

## 1. Your File Ownership

You only ever edit **your files**. Never touch files outside your area.

| Role | Branch | Your files | Your job |
|---|---|---|---|
| **Prediction A** | `pred-A/indicators` | `prediction/indicators.py` | Fetch historical data from yfinance, compute all technical indicators (RSI, SMA, MACD, Volatility, Volume Change), return a clean DataFrame with a `signal` column |
| **Prediction B** | `pred-B/model` | `prediction/model.py` `prediction/train.py` | Train the Random Forest classifier on A's DataFrame using TimeSeriesSplit, save the model as `prediction/saved_models/signal_model.pkl`, expose `predict_signal(ticker)` |
| **Prediction C** | `pred-C/signal-engine` | `prediction/signal_engine.py` | Wire A and B together into the single public function `get_signal(ticker)` that the rest of the system calls — must match the contract in Section 8 exactly |
| **RAG Teammate** | `teammate/rag-pipeline` | `rag/retriever.py` `rag/ingest.py` `pipeline/online_rag.py` `data/documents/` | Populate the knowledge base with finance documents, tune retrieval parameters, and upgrade the online RAG session store |
| **Adham** | `main` + feature branches | Everything else | Core architecture, pipeline, API, market module, frontend logic, integration |

If you are unsure whether a file is yours — ask Adham before editing it.

---

## 2. Branch Structure

The project uses a layered branch structure to keep `main` clean and prevent conflicts.

```
main                            ← protected, Adham merges here only
    │
    ├── team/prediction         ← prediction integration branch
    │       │                      all three merge their branches here first
    │       ├── pred-A/indicators
    │       ├── pred-B/model
    │       └── pred-C/signal-engine
    │
    └── teammate/rag-pipeline   ← RAG teammate, PRs directly to main
```

**Prediction team:** Each person works alone on their own branch. When a piece is complete, they open a PR into `team/prediction` — not into `main`. Adham merges `team/prediction` into `main` only when the entire prediction module is working end-to-end.

**RAG teammate:** Works alone on `teammate/rag-pipeline`. PRs directly into `main` when done.

**Adham:** Works on `main` directly or opens short-lived feature branches for larger changes (e.g. `adham/context-builder-signals`) and merges them himself.

---

## 3. One-Time Setup

Do this once on your machine.

```bash
# 1. Clone the repo
git clone https://github.com/your-username/X-Invest.git
cd X-Invest

# 2. Switch to your branch
# Prediction A:
git checkout pred-A/indicators
# Prediction B:
git checkout pred-B/model
# Prediction C:
git checkout pred-C/signal-engine
# RAG Teammate:
git checkout teammate/rag-pipeline

# 3. Create your virtual environment
python -m venv .venv

# 4. Activate it
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate

# 5. Install dependencies
pip install -r requirements.txt

# 6. Copy the environment file
copy .env.example .env
```

You are now set up. You only do this once.

---

## 4. Daily Workflow

Every time you sit down to work, follow these steps in order.

```bash
# Step 1 — Pull the latest changes on your branch before starting
git pull origin your-branch-name

# Step 2 — Do your work (edit your files only)

# Step 3 — Stage your changes
git add .

# Step 4 — Commit with a proper message (see Section 5)
git commit -m "feat: add RSI indicator calculation"

# Step 5 — Push to your branch
git push origin your-branch-name
```

**Commit and push at least once per session.** Do not wait until everything is done to push for the first time.

---

## 5. Commit Message Convention

Every commit message follows this format:

```
<type>: <short description>
```

- Maximum 72 characters
- Present tense ("add" not "added")
- No period at the end

### Types

| Type | Use it when |
|---|---|
| `feat:` | you added new functionality |
| `fix:` | you fixed a bug |
| `refactor:` | you restructured code without changing behaviour |
| `docs:` | you updated a comment, README, or document |
| `chore:` | config change, added a file, cleanup |

### Examples per role

**Prediction A**
```bash
git commit -m "feat: add RSI and SMA crossover indicators"
git commit -m "feat: add MACD and Volatility columns to get_features()"
git commit -m "fix: handle empty DataFrame when ticker has no history"
git commit -m "refactor: extract FEATURES list to module-level constant"
```

**Prediction B**
```bash
git commit -m "feat: train Random Forest classifier with TimeSeriesSplit"
git commit -m "feat: save trained model to prediction/saved_models/"
git commit -m "fix: use TimeSeriesSplit instead of random train_test_split"
git commit -m "feat: expose predict_signal() inference function in model.py"
```

**Prediction C**
```bash
git commit -m "feat: implement get_signal() in signal_engine.py"
git commit -m "fix: return unknown signal instead of raising on bad ticker"
git commit -m "feat: add disclaimer field to signal response dict"
```

**RAG Teammate**
```bash
git commit -m "docs: add CFA Level 1 summary PDF to knowledge base"
git commit -m "docs: add EGX official guide and Investopedia export"
git commit -m "chore: run rag ingest after adding finance documents"
git commit -m "feat: tune MMR lambda and pool size for better retrieval"
git commit -m "feat: upgrade online_rag to semantic session search"
```

---

## 6. How the Prediction Team Works in Parallel

All three people start on the same day — nobody waits for anyone else. This is done using stubs.

**Day one:** Before writing real code, each person puts a stub in their file so the others can import from it immediately and start working.

**Prediction A** puts this stub in `indicators.py` so B can start right away:
```python
def get_features(ticker: str) -> pd.DataFrame:
    # STUB — replace with real implementation
    import numpy as np
    dates = pd.date_range("2024-01-01", periods=100, freq="B")
    return pd.DataFrame({
        "SMA_cross":     np.random.randint(0, 2, 100),
        "RSI":           np.random.uniform(30, 70, 100),
        "MACD":          np.random.uniform(-2, 2, 100),
        "Volatility":    np.random.uniform(0.01, 0.05, 100),
        "Volume_Change": np.random.uniform(-0.2, 0.2, 100),
        "signal":        np.random.choice(["bullish","neutral","bearish"], 100),
    }, index=dates)
```

**Prediction B** puts this stub in `model.py` so C can start right away:
```python
def predict_signal(ticker: str) -> dict:
    # STUB — replace with real implementation
    return {
        "signal":     "bullish",
        "confidence": 65.0,
        "rsi":        55.0,
        "sma_cross":  True,
    }
```

When A finishes the real implementation and merges into `team/prediction`, B pulls and the real code replaces the stub automatically — same import, no rewrite needed. When B finishes, C does the same.

### Merging into team/prediction

When your piece is complete and tested, open a PR into `team/prediction` (see Section 7). Merge order matters:

```
1. pred-A/indicators     → team/prediction   (first)
2. pred-B/model          → team/prediction   (second, after A merges)
3. pred-C/signal-engine  → team/prediction   (third, after B merges)
4. team/prediction       → main              (Adham does this last)
```

---

## 7. Opening a Pull Request (when you are done)

When your work is complete and tested, open a Pull Request on GitHub.
This tells Adham your branch is ready to be reviewed.

```
1. Go to github.com/your-username/X-Invest
2. Click the banner "Your branch had recent pushes" → Compare & pull request
   OR go to Pull Requests tab → New pull request
3. Prediction team:  base: team/prediction  ←  compare: your-branch
   RAG teammate:     base: main             ←  compare: teammate/rag-pipeline
4. Fill in the title and description using the template below
5. Click "Create Pull Request"
6. Wait for Adham to review — he may leave comments asking for changes
7. Fix any comments → commit → push (the PR updates automatically)
8. Adham approves → clicks Merge → you are done
```

### PR Title

Same format as a commit message:

```
feat: prediction indicators (RSI, SMA, MACD, Volatility)
```

### PR Description Template

Copy this and fill it in every time:

```markdown
## What this does
Short explanation of what you built and which files you changed.

## How to test it
Step-by-step instructions so Adham can verify it works.
Example:
  python prediction/train.py
  → should print classification report and save signal_model.pkl

## Files changed
- prediction/indicators.py
- prediction/model.py

## Notes
Any decisions you made, edge cases you handled, or things still missing.
```

---

## 8. The Contract for Prediction C (signal_engine.py)

The rest of the pipeline already calls `get_signal()`. Your function **must** match this exact structure or the API will break:

```python
def get_signal(ticker: str) -> dict:
    return {
        "ticker":     "AAPL",
        "signal":     "bullish",   # must be: bullish | neutral | bearish | unknown
        "confidence": 72.3,        # float 0–100
        "rsi":        58.4,        # float or None
        "sma_cross":  True,        # bool or None
        "rf_signal":  "bullish",   # str or None
        "disclaimer": "...",       # non-empty string
        "error":      ""           # empty string if no error
    }
```

**Critical rule: this function must never raise an exception.** Catch all errors internally and return `"signal": "unknown"` with the error message in the `"error"` field.

### How to test Prediction C

Call `get_signal()` for several tickers including a bad one:

```python
from prediction.signal_engine import get_signal

for ticker in ["AAPL", "TSLA", "MSFT", "INVALID_TICKER"]:
    result = get_signal(ticker)
    print(f"{ticker}: {result}")

# AAPL:           {"signal": "bullish", "confidence": 71.2, "error": "", ...}
# TSLA:           {"signal": "neutral", "confidence": 58.4, "error": "", ...}
# MSFT:           {"signal": "bearish", "confidence": 63.1, "error": "", ...}
# INVALID_TICKER: {"signal": "unknown", "confidence": 0,    "error": "no data found", ...}
# INVALID_TICKER must not crash — error goes in the dict, never raised as exception
```

### How to test Prediction B

Run the training script and verify the output:

```bash
python prediction/train.py
# should print a classification report like:
#
#               precision    recall  f1-score
#    bearish       0.48      0.41      0.44
#    bullish       0.51      0.58      0.54
#    neutral       0.44      0.46      0.45
#
# Model saved to prediction/saved_models/signal_model.pkl
```

Note: 50–55% accuracy across 3 classes is normal and expected for stock prediction.
If you see 90%+ accuracy something is wrong — you are likely using random split instead
of TimeSeriesSplit, which leaks future data into training.

### How to test Prediction A

```python
from prediction.indicators import get_features

df = get_features("AAPL")
print(df.columns.tolist())
# ["SMA_20", "SMA_50", "SMA_cross", "RSI", "MACD", "Volatility", "Volume_Change", "signal"]

print(df.tail(5))
# last 5 rows, no NaN values

print(df["signal"].value_counts())
# should show a mix of bullish / neutral / bearish — not all one class
```

---

## 9. The Contract for the RAG Teammate (rag pipeline)

You own three things and they are independent — you can work on all three in any order.

### 9a. Knowledge base documents (data/documents/)

Drop any `.pdf`, `.docx`, `.txt`, or `.md` finance document into `data/documents/`, then run:

```bash
python -m rag.ingest
```

That command reads every file, chunks it into 400-word overlapping pieces, embeds each chunk via Ollama, and stores everything in ChromaDB. The chat pipeline picks it up automatically on the next question.

**Target: 20+ quality documents.** Suggested content:
- Financial textbooks (PDF)
- Investopedia articles saved as PDF or TXT
- EGX (Egyptian Exchange) official guides
- CFA Level 1 summary notes
- Arabic finance reference material

**How to verify:** Start the server, ask "ما هو مؤشر P/E؟" and check that the response contains specific content from your documents — not just the model's own knowledge.

### 9b. Retrieval quality tuning (rag/retriever.py)

After adding documents, test retrieval quality and tune these parameters:

```python
pool_size = min(20, col.count())  # candidate pool for MMR — increase if results are poor
lambda_   = 0.5                   # 0 = pure diversity, 1 = pure relevance
n_results = 3                     # chunks returned — increase to 5 if context is too thin
# distance threshold 1.2          # lower = stricter relevance filter
```

Test with Arabic and English queries. If retrieved chunks are irrelevant, tighten the threshold. If results are too repetitive, lower lambda toward 0. Document which values you tested and why you chose the final ones.

### 9c. Online RAG upgrade (pipeline/online_rag.py)

Currently the session store returns all stored strings concatenated. The upgrade is to add proper semantic search within the session — so follow-up questions retrieve the most relevant previously-fetched data rather than dumping everything at once.

**How to verify:** Ask about Apple, then ask a follow-up with no ticker mentioned. The response should use the Apple data from the session without the user having to mention Apple again.

---

## 10. Rules

1. **Never push directly to `main`.** Your branch only. Adham merges.
2. **Never edit files outside your ownership table** in Section 1.
3. **Never commit your `.env` file.** It is in `.gitignore` for a reason — it contains local settings.
4. **Commit often.** Small commits are easier to review and easier to undo if something breaks.
5. **Pull before you start working** every session. This keeps your branch up to date.
6. **Prediction team:** your PRs go to `team/prediction`, not to `main`.

---

## 11. If Something Goes Wrong

**"I accidentally edited the wrong file"**
```bash
git checkout -- path/to/wrong/file   # discard changes to that file
```

**"I pushed something broken"**
Tell Adham immediately. Do not try to force-push or rewrite history.

**"I have a merge conflict"**
Stop. Message Adham. He will resolve it — this is not something you need to handle alone.

**"I don't know if my code is correct"**
Push what you have and open a Draft Pull Request. Adham can review it early and give feedback before it is marked ready.

---

*Questions? Message Adham directly.*
