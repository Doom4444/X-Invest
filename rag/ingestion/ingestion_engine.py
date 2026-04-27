import os

from rag.ingestion.file_router import load_file
from rag.preprocessing.cleaner import clean_text
from rag.preprocessing.chunker import chunk_text
from rag.retrieval.vector_store import VectorStore


def ingest_folder(folder_path: str):

    vector_store = VectorStore()

    for file in os.listdir(folder_path):

        file_path = os.path.join(folder_path, file)

        if not os.path.isfile(file_path):
            continue

        print(f"\nProcessing: {file}")

        raw_text = load_file(file_path)

        cleaned_text = clean_text(raw_text)

        chunks = chunk_text(cleaned_text)

        print(f"Chunks created: {len(chunks)}")

        vector_store.add_chunks(chunks)