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

            # --------------------------------
            # Cosine similarity conversion
            # --------------------------------
            similarity = max(
                0,
                1 - dist
            )

            # --------------------------------
            # Keyword overlap
            # --------------------------------
            keyword_overlap = (
                self.keyword_score(
                    query,
                    doc
                )
            )

            keyword_overlap = min(
                keyword_overlap / 10,
                1
            )

            # --------------------------------
            # Final score
            # --------------------------------
            final_score = (

                (0.75 * similarity) +

                (0.25 * keyword_overlap)
            )

            reranked.append({

                "document": doc,

                "distance": dist,

                "similarity": similarity,

                "score": final_score
            })

        # -----------------------------------
        # Sort descending
        # -----------------------------------
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

        similarities = [

            item["similarity"]

            for item in reranked
        ]

        return (
            documents,
            distances,
            similarities
        )

    # -----------------------------------
    # Retrieve relevant documents
    # -----------------------------------
    def retrieve(
        self,
        query: str,
        top_k: int = 5
    ):

        try:

            # --------------------------------
            # Vector search
            # --------------------------------
            results = self.vector_store.search(

                query,

                top_k * 3
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

            # --------------------------------
            # Clean invalid docs
            # --------------------------------
            clean_documents = []

            clean_distances = []

            for doc, dist in zip(
                documents,
                distances
            ):

                if (
                    doc and
                    doc.strip()
                ):

                    clean_documents.append(
                        doc
                    )

                    clean_distances.append(
                        dist
                    )

            # --------------------------------
            # Empty retrieval
            # --------------------------------
            if not clean_documents:

                print(
                    "[Retriever] "
                    "No documents found"
                )

                return [], []

            # --------------------------------
            # Reranking
            # --------------------------------
            (
                reranked_documents,
                reranked_distances,
                reranked_similarities

            ) = self.rerank(

                query,

                clean_documents,

                clean_distances
            )

            # --------------------------------
            # Confidence filtering
            # --------------------------------
            final_documents = []

            final_distances = []

            for doc, dist, sim in zip(

                reranked_documents,

                reranked_distances,

                reranked_similarities
            ):

                print(

                    f"[Retriever] "

                    f"Distance: {round(dist, 3)} | "

                    f"Similarity: {round(sim, 3)}"
                )

                # --------------------------------
                # Realistic threshold
                # --------------------------------
                if sim >= 0.55:

                    final_documents.append(
                        doc
                    )

                    final_distances.append(
                        dist
                    )

            # --------------------------------
            # Final fallback
            # --------------------------------
            if not final_documents:

                print(
                    "[Retriever] "
                    "No relevant chunks"
                )

                return [], []

            # --------------------------------
            # Return top_k
            # --------------------------------
            return (

                final_documents[:top_k],

                final_distances[:top_k]
            )

        except Exception as e:

            print(
                f"[Retriever] Error: {e}"
            )

            return [], []