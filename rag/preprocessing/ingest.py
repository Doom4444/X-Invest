import os
import uuid

from rag.preprocessing.utils import load_document, clean_text, chunk_text
from rag.core.vector_store import VectorStore


def ingest_folder(folder_path: str):

    vector_store = VectorStore()

    for file_name in os.listdir(folder_path):

        file_path = os.path.join(folder_path, file_name)

        if not os.path.isfile(file_path):
            continue

        print(f"\nProcessing: {file_name}")

        # 1. load file
        text = load_document(file_path)

        if not text:
            print(f"[Skipped] No text extracted: {file_name}")
            continue

        # 2. clean text 🔥
        text = clean_text(text)

        # 3. chunk with overlap 🔥
        chunks = chunk_text(text)

        if not chunks:
            print(f"[Skipped] No chunks created: {file_name}")
            continue

        print(f"Chunks created: {len(chunks)}")

        # 4. IDs + metadata
        ids = [str(uuid.uuid4()) for _ in chunks]
        metadatas = [{"source": file_name} for _ in chunks]

        # 5. store
        vector_store.add_documents(ids, chunks, metadatas)

    print("\n✅ Ingestion completed.")