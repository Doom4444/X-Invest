from rag.core.vector_store import VectorStore


class Retriever:

    def __init__(self):

        self.vector_store = VectorStore()

    # -----------------------------------
    # Retrieve relevant documents
    # -----------------------------------
    def retrieve(
        self,
        query: str,
        top_k: int = 5
    ):

        try:

            # -----------------------------
            # Vector search
            # -----------------------------
            results = self.vector_store.search(
                query,
                top_k
            )

            documents = (
                results.get("documents", [[]])[0]
            )

            distances = (
                results.get("distances", [[]])[0]
            )

            # -----------------------------
            # Clean invalid docs
            # -----------------------------
            clean_documents = []

            clean_distances = []

            for doc, dist in zip(
                documents,
                distances
            ):

                if doc and doc.strip():

                    clean_documents.append(doc)

                    clean_distances.append(dist)

            return (
                clean_documents,
                clean_distances
            )

        except Exception as e:

            print(
                f"[Retriever] Error: {e}"
            )

            return [], []