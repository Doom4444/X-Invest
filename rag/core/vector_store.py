import chromadb
from rag.core.embeddings import EmbeddingModel


class VectorStore:

    def __init__(self):

        # 🔥 أهم تعديل
        self.client = chromadb.PersistentClient(path="db/chroma")

        self.embedding_model = EmbeddingModel()

        self.collection = self.client.get_or_create_collection(
            name="finance_documents"
        )

        print("Collection count:", self.collection.count())

    # -----------------------------
    def add_documents(self, ids, docs, metadatas=None):

        if metadatas is None:
            metadatas = [{} for _ in docs]

        embeddings = self.embedding_model.embed(docs)

        self.collection.add(
            ids=ids,
            documents=docs,
            embeddings=embeddings,
            metadatas=metadatas
        )

        print(f"Stored {len(docs)} chunks in vector database")

    # -----------------------------
    def search(self, query: str, top_k: int = 5):

        query_embedding = self.embedding_model.embed([query])[0]

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        return results