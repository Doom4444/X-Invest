import chromadb

from rag.core.embeddings import EmbeddingModel


class VectorStore:

    def __init__(self):

        # -----------------------------------
        # Persistent Chroma Client
        # -----------------------------------
        self.client = chromadb.PersistentClient(
            path="db/chroma"
        )

        # -----------------------------------
        # Embedding Model
        # -----------------------------------
        self.embedding_model = (
            EmbeddingModel()
        )

        # -----------------------------------
        # Collection
        # IMPORTANT:
        # Use cosine similarity
        # -----------------------------------
        self.collection = (

            self.client.get_or_create_collection(

                name="finance_documents",

                metadata={
                    "hnsw:space": "cosine"
                }
            )
        )

        print(

            "Collection count:",

            self.collection.count()
        )

    # -----------------------------------
    # Add Documents
    # -----------------------------------
    def add_documents(

        self,

        ids,

        docs,

        metadatas=None
    ):

        if metadatas is None:

            metadatas = [

                {}

                for _ in docs
            ]

        # -----------------------------------
        # Generate embeddings
        # -----------------------------------
        embeddings = (
            self.embedding_model.embed(
                docs
            )
        )

        # -----------------------------------
        # Store in Chroma
        # -----------------------------------
        self.collection.add(

            ids=ids,

            documents=docs,

            embeddings=embeddings,

            metadatas=metadatas
        )

        print(

            f"Stored {len(docs)} "

            f"chunks in vector database"
        )

    # -----------------------------------
    # Search
    # -----------------------------------
    def search(
        self,
        query: str,
        top_k: int = 5
    ):

        # -----------------------------------
        # Query embedding
        # -----------------------------------
        query_embedding = (

            self.embedding_model.embed(
                [query]
            )[0]
        )

        # -----------------------------------
        # Chroma query
        # -----------------------------------
        results = self.collection.query(

            query_embeddings=[
                query_embedding
            ],

            n_results=top_k
        )

        return results