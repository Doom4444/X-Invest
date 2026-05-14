import re

from rag.core.vector_store import VectorStore


class Retriever:

    def __init__(self):

        self.vector_store = VectorStore()

    # -----------------------------------
    # Keyword overlap score
    # -----------------------------------
    def keyword_score(
        self,
        query,
        document
    ):

        query_words = set(

            re.findall(
                r"\w+",
                query.lower()
            )
        )

        doc_words = set(

            re.findall(
                r"\w+",
                document.lower()
            )
        )

        overlap = (
            query_words & doc_words
        )

        return len(overlap)

    # -----------------------------------
    # Lightweight reranking
    # -----------------------------------
    def rerank(
        self,
        query,
        documents,
        distances
    ):

        reranked = []

        for doc, dist in zip(
            documents,
            distances
        ):

            # -----------------------------
            # Embedding similarity
            # -----------------------------
            similarity = (
                1 / (1 + dist)
            )

            # -----------------------------
            # Keyword overlap
            # -----------------------------
            keyword_overlap = (
                self.keyword_score(
                    query,
                    doc
                )
            )

            # normalize overlap
            keyword_overlap = min(
                keyword_overlap / 10,
                1
            )

            # -----------------------------
            # Final score
            # -----------------------------
            final_score = (

                (0.7 * similarity) +

                (0.3 * keyword_overlap)
            )

            reranked.append({

                "document": doc,

                "distance": dist,

                "score": final_score
            })

        # sort descending
        reranked.sort(

            key=lambda x: x["score"],

            reverse=True
        )

        documents = [

            item["document"]

            for item in reranked
        ]

        distances = [

            item["distance"]

            for item in reranked
        ]

        return documents, distances

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

                top_k * 2
            )

            documents = (
                results.get(
                    "documents",
                    [[]]
                )[0]
            )

            distances = (
                results.get(
                    "distances",
                    [[]]
                )[0]
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

            # -----------------------------
            # Reranking
            # -----------------------------
            reranked_documents, reranked_distances = (

                self.rerank(

                    query,

                    clean_documents,

                    clean_distances
                )
            )

            # -----------------------------
            # Return final top_k
            # -----------------------------
            return (

                reranked_documents[:top_k],

                reranked_distances[:top_k]
            )

        except Exception as e:

            print(
                f"[Retriever] Error: {e}"
            )

            return [], []