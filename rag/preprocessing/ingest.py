import os
import uuid

from rag.preprocessing.utils import load_document, clean_text, chunk_text
from rag.core.vector_store import VectorStore
from config import COLLECTION_NAME


def ingest_folder(folder_path: str):

    vector_store = VectorStore()
    
    # Wipe and recreate so re-ingest is always clean
    vector_store.client.delete_collection(COLLECTION_NAME)
    vector_store.collection = vector_store.client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    for file_name in os.listdir(folder_path):
        
        # Skip hidden files and placeholders
        if file_name.startswith(".") or file_name == ".gitkeep":
            continue

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