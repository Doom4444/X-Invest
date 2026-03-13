# rag/ingest.py
#
# WHAT IT DOES:
# Reads every document in knowledge_base/docs/, splits them into chunks,
# embeds each chunk via nomic-embed-text (Ollama), and stores in ChromaDB.
#
# SUPPORTED FILE TYPES:
#   .pdf   → extracted via pdfplumber (handles multi-column layouts better than pypdf)
#   .docx  → extracted via python-docx
#   .txt   → read directly
#   .md    → read directly (strips markdown symbols before embedding)
#
# HOW TO USE:
#   1. Drop your documents into knowledge_base/docs/
#   2. Run: python -m knowledge_base.ingest
#   3. ChromaDB is rebuilt from scratch every time
#   4. After this, rag_retriever.py can query the collection
#
# CHUNKING STRATEGY:
#   Target: ~400 words per chunk with 50-word overlap between chunks.
#   Why overlap? So a concept that spans a paragraph boundary doesn't
#   get cut in half and become unretrievable.
#
# WHAT DOCUMENTS TO ADD (knowledge_base/docs/):
#   - Financial textbooks (PDF)
#   - Investopedia exports or similar finance reference pages (saved as PDF/TXT)
#   - EGX (Egyptian Exchange) official guides
#   - CFA Level 1 summary notes
#   - Any Arabic finance reference material
#   The more documents you add, the richer the RAG answers become.

import os, re, chromadb, requests
from pathlib import Path
from config import OLLAMA_URL, EMBED_MODEL, CHROMA_PATH, COLLECTION_NAME, DOCS_PATH

DOCS_DIR    = Path(DOCS_PATH)
CHUNK_SIZE  = 400   # words per chunk
CHUNK_OVERLAP = 50  # words of overlap between consecutive chunks


# ── Document Readers ──────────────────────────────────────────────────────────

def read_pdf(path: Path) -> str:
    """Extract text from PDF using pdfplumber (handles tables + columns)."""
    try:
        import pdfplumber
        text_parts = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    text_parts.append(text)
        return "\n\n".join(text_parts)
    except Exception as e:
        print(f"  [PDF error] {path.name}: {e}")
        return ""

def read_docx(path: Path) -> str:
    """Extract text from DOCX file."""
    try:
        from docx import Document
        doc = Document(path)
        return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
    except Exception as e:
        print(f"  [DOCX error] {path.name}: {e}")
        return ""

def read_txt(path: Path) -> str:
    """Read plain text file, try UTF-8 then fallback to latin-1."""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="latin-1")

def read_md(path: Path) -> str:
    """Read markdown and strip symbols so embeddings focus on content."""
    text = read_txt(path)
    text = re.sub(r"#{1,6}\s+", "", text)       # remove headings
    text = re.sub(r"\*{1,2}(.+?)\*{1,2}", r"\1", text)  # bold/italic
    text = re.sub(r"`{1,3}[^`]*`{1,3}", "", text)        # code blocks
    text = re.sub(r"\[(.+?)\]\(.+?\)", r"\1", text)      # links
    return text

def read_file(path: Path) -> str:
    """Dispatch to the right reader based on file extension."""
    ext = path.suffix.lower()
    if ext == ".pdf":   return read_pdf(path)
    if ext == ".docx":  return read_docx(path)
    if ext in (".txt", ".md"): return read_txt(path) if ext == ".txt" else read_md(path)
    print(f"  Skipping unsupported file type: {path.name}")
    return ""


# ── Chunker ───────────────────────────────────────────────────────────────────

def chunk_text(text: str, source: str) -> list[dict]:
    """
    Split text into overlapping word-window chunks.

    Returns list of:
        { "text": str, "source": filename, "chunk_index": int }

    Why word-based not character-based?
    Words map more directly to semantic content.
    400 words ≈ 600-800 tokens, which fits comfortably in nomic-embed-text's
    8192 token context window with room to spare.
    """
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split()

    if not words:
        return []

    chunks = []
    start  = 0
    idx    = 0

    while start < len(words):
        end        = min(start + CHUNK_SIZE, len(words))
        chunk_text = " ".join(words[start:end])

        # Skip chunks that are too short to be meaningful
        if len(chunk_text) > 50:
            chunks.append({
                "text":        chunk_text,
                "source":      source,
                "chunk_index": idx,
            })
            idx += 1

        if end == len(words):
            break
        start = end - CHUNK_OVERLAP  # overlap: step back before next chunk

    return chunks


# ── Embedder ──────────────────────────────────────────────────────────────────

def embed(text: str) -> list[float]:
    """Get embedding vector from Ollama nomic-embed-text."""
    r = requests.post(
        f"{OLLAMA_URL}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=30
    )
    r.raise_for_status()
    return r.json()["embedding"]


# ── Main Ingest Pipeline ──────────────────────────────────────────────────────

def ingest() -> None:
    # ── 0. Scan docs folder ───────────────────────────────────────────────────
    supported = {".pdf", ".docx", ".txt", ".md"}
    doc_files = [f for f in DOCS_DIR.iterdir()
                 if f.is_file() and f.suffix.lower() in supported]

    if not doc_files:
        print(f"No documents found in {DOCS_DIR}/")
        print("Add .pdf, .docx, .txt, or .md files and run again.")
        return

    print(f"Found {len(doc_files)} document(s): {[f.name for f in doc_files]}")

    # ── 1. Test Ollama connection ─────────────────────────────────────────────
    print(f"\nTesting Ollama ({EMBED_MODEL})...", end=" ", flush=True)
    try:
        vec = embed("connection test")
        print(f"OK — vector size: {len(vec)}")
    except Exception as e:
        print(f"FAILED\nCannot reach Ollama: {e}")
        print("Make sure Ollama is running: ollama serve")
        print("And the model is pulled:    ollama pull nomic-embed-text:latest")
        return

    # ── 2. Read + chunk all documents ─────────────────────────────────────────
    all_chunks = []
    for doc_path in doc_files:
        print(f"\nReading: {doc_path.name}")
        raw_text = read_file(doc_path)
        if not raw_text.strip():
            print(f"  Warning: no text extracted from {doc_path.name}")
            continue
        chunks = chunk_text(raw_text, source=doc_path.name)
        print(f"  Extracted {len(raw_text.split())} words → {len(chunks)} chunks")
        all_chunks.extend(chunks)

    if not all_chunks:
        print("\nNo chunks to ingest. Check that your documents have extractable text.")
        return

    print(f"\nTotal chunks to embed: {len(all_chunks)}")

    # ── 3. Embed all chunks ───────────────────────────────────────────────────
    docs, ids, metas, embeddings = [], [], [], []
    failed = 0

    for i, chunk in enumerate(all_chunks):
        print(f"  [{i+1}/{len(all_chunks)}] {chunk['source']} chunk {chunk['chunk_index']}...",
              end=" ", flush=True)
        try:
            emb = embed(chunk["text"])
            docs.append(chunk["text"])
            ids.append(f"chunk_{i}")
            metas.append({
                "source":      chunk["source"],
                "chunk_index": chunk["chunk_index"],
            })
            embeddings.append(emb)
            print("OK")
        except Exception as e:
            print(f"FAILED ({e})")
            failed += 1

    # ── 4. Store in ChromaDB ──────────────────────────────────────────────────
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        client.delete_collection(COLLECTION_NAME)
        print("\nDeleted old ChromaDB collection.")
    except Exception:
        pass

    col = client.create_collection(COLLECTION_NAME)
    col.add(documents=docs, embeddings=embeddings, ids=ids, metadatas=metas)

    print(f"\nDone! Ingested {len(docs)} chunks from {len(doc_files)} document(s).")
    if failed:
        print(f"Warning: {failed} chunk(s) failed to embed.")
    print("RAG pipeline is now active.")
    print(f"ChromaDB collection '{COLLECTION_NAME}' at: {CHROMA_PATH}")

if __name__ == "__main__":
    ingest()
