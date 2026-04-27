from rag.retrieval.vector_store import VectorStore
from rag.retrieval.embeddings import EmbeddingModel


class Retriever:

    def __init__(self):

        self.vector_store = VectorStore()
        self.embedding_model = EmbeddingModel()

    def search(self, query, top_k=5):

        
        query_embedding = self.embedding_model.embed([query])[0]

        
        results = self.vector_store.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        documents = results["documents"][0]

        return documents