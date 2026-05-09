# rag/retriever.py
from rag.core.vector_store import VectorStore


class Retriever:

    def __init__(self):
        self.vector_store = VectorStore()

    def retrieve(self, query: str, top_k: int = 5):

        # search using vector store
        results = self.vector_store.search(query, top_k)

        documents = results["documents"][0]
        distances = results["distances"][0]
        documents = [doc for doc in documents if doc]

        return documents, distances