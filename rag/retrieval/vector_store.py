import chromadb
from rag.retrieval.embeddings import EmbeddingModel


class VectorStore:

    def __init__(self):

        self.client = chromadb.Client(
            chromadb.config.Settings(
                persist_directory="db/chroma"
            )
        )

        self.embedding_model = EmbeddingModel()

        self.collection = self.client.get_or_create_collection(
            name="finance_documents"
        )

    def add_chunks(self, chunks):

        embeddings = self.embedding_model.embed(chunks)

        ids = [str(i) for i in range(len(chunks))]

        self.collection.add(
            documents=chunks,
            embeddings=embeddings,
            ids=ids
        )

        print(f"Stored {len(chunks)} chunks in vector database")