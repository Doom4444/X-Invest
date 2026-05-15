import re

from rank_bm25 import BM25Okapi

from rag.core.vector_store import VectorStore


class Retriever:

    def __init__(self):

        self.vector_store = VectorStore()

    # -----------------------------------
    # Tokenize text
    # -----------------------------------
    def tokenize(
        self,
        text
    ):

        return re.findall(
            r"\w+",
            text.lower()
        )

    # -----------------------------------
    # BM25 Scoring
    # -----------------------------------
    def bm25_scores(

        self,

        query,

        documents
    ):

        tokenized_docs = [

            self.tokenize(doc)

            for doc in documents
        ]

        bm25 = BM25Okapi(
            tokenized_docs
        )

        tokenized_query = (
            self.tokenize(query)
        )

        scores = bm25.get_scores(
            tokenized_query
        )

        # --------------------------------
        # Normalize scores
        # --------------------------------
        max_score = max(scores)

        if max_score == 0:

            return [0] * len(scores)

        normalized_scores = [

            s / max_score

            for s in scores
        ]

        return normalized_scores

    # -----------------------------------
    # Hybrid reranking
    # -----------------------------------
    def rerank(

        self,

        query,

        documents,

        distances
    ):

        reranked = []

        # --------------------------------
        # BM25 scores
        # --------------------------------
        bm25_scores = self.bm25_scores(

            query,

            documents
        )

        for (
            doc,
            dist,
            bm25_score

        ) in zip(

            documents,

            distances,

            bm25_scores
        ):

            # --------------------------------
            # Cosine similarity conversion
            # --------------------------------
            semantic_similarity = max(

                0,

                1 - dist
            )

            # --------------------------------
            # Hybrid final score
            # --------------------------------
            final_score = (

                (0.70 * semantic_similarity) +

                (0.30 * bm25_score)
            )

            reranked.append({

                "document": doc,

                "distance": dist,

                "semantic_similarity":
                    semantic_similarity,

                "bm25_score":
                    bm25_score,

                "score":
                    final_score
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

            item["semantic_similarity"]

            for item in reranked
        ]

        return (

            documents,

            distances,

            similarities
        )

    # -----------------------------------
    # Remove duplicate chunks
    # -----------------------------------
    def deduplicate(

        self,

        documents,

        distances,

        similarity_threshold=0.90
    ):

        unique_documents = []

        unique_distances = []

        for doc, dist in zip(
            documents,
            distances
        ):

            is_duplicate = False

            current_words = set(

                self.tokenize(doc)
            )

            for existing_doc in unique_documents:

                existing_words = set(

                    self.tokenize(
                        existing_doc
                    )
                )

                # ----------------------------
                # Jaccard similarity
                # ----------------------------
                intersection = len(

                    current_words &
                    existing_words
                )

                union = len(

                    current_words |
                    existing_words
                )

                if union == 0:

                    continue

                similarity = (
                    intersection / union
                )

                # ----------------------------
                # Duplicate detected
                # ----------------------------
                if similarity >= similarity_threshold:

                    is_duplicate = True

                    break

            # ----------------------------
            # Keep unique chunk
            # ----------------------------
            if not is_duplicate:

                unique_documents.append(
                    doc
                )

                unique_distances.append(
                    dist
                )

        return (
            unique_documents,
            unique_distances
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
            # Hybrid reranking
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
            # Deduplicate chunks
            # --------------------------------
            (
                final_documents,
                final_distances

            ) = self.deduplicate(

                final_documents,

                final_distances,

                similarity_threshold=0.90
            )

            print(

                f"[Retriever] "

                f"Final unique chunks: "

                f"{len(final_documents)}"
            )

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