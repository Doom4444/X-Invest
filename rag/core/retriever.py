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
    # Dynamic similarity threshold
    # -----------------------------------
    def get_similarity_threshold(
        self,
        intent
    ):

        threshold_map = {

            # --------------------------------
            # Precise retrieval
            # --------------------------------
            "forecast": 0.60,

            "market_data": 0.60,

            # --------------------------------
            # Broader retrieval
            # --------------------------------
            "analysis": 0.50,

            "general_finance": 0.45
        }

        return threshold_map.get(
            intent,
            0.55
        )

    # -----------------------------------
    # Dynamic top_k
    # -----------------------------------
    def get_top_k(
        self,
        intent
    ):

        topk_map = {

            "forecast": 2,

            "market_data": 2,

            "analysis": 5,

            "general_finance": 6
        }

        return topk_map.get(
            intent,
            3
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

        return reranked

    # -----------------------------------
    # Remove duplicate chunks
    # -----------------------------------
    def deduplicate(

        self,

        reranked_items,

        similarity_threshold=0.90
    ):

        unique_items = []

        for item in reranked_items:

            current_doc = item[
                "document"
            ]

            is_duplicate = False

            current_words = set(

                self.tokenize(
                    current_doc
                )
            )

            for existing in unique_items:

                existing_words = set(

                    self.tokenize(

                        existing[
                            "document"
                        ]
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
            # Keep best unique chunk
            # ----------------------------
            if not is_duplicate:

                unique_items.append(
                    item
                )

        return unique_items

    # -----------------------------------
    # Retrieve relevant documents
    # -----------------------------------
    def retrieve(

        self,

        query: str,

        intent="general_finance"
    ):

        try:

            # --------------------------------
            # Dynamic retrieval config
            # --------------------------------
            top_k = self.get_top_k(
                intent
            )

            similarity_threshold = (

                self.get_similarity_threshold(
                    intent
                )
            )

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
            reranked_items = self.rerank(

                query,

                clean_documents,

                clean_distances
            )

            # --------------------------------
            # Confidence filtering
            # --------------------------------
            filtered_items = []

            for item in reranked_items:

                similarity = item[
                    "semantic_similarity"
                ]

                distance = item[
                    "distance"
                ]

                print(

                    f"[Retriever] "

                    f"Distance: "
                    f"{round(distance, 3)} | "

                    f"Similarity: "
                    f"{round(similarity, 3)}"
                )

                # --------------------------------
                # Dynamic threshold
                # --------------------------------
                if similarity >= similarity_threshold:

                    filtered_items.append(
                        item
                    )

            # --------------------------------
            # Fallback retrieval
            # --------------------------------
            if not filtered_items:

                print(
                    "[Retriever] "
                    "Using fallback chunk"
                )

                filtered_items = [
                    reranked_items[0]
                ]

            # --------------------------------
            # Deduplicate chunks
            # --------------------------------
            filtered_items = self.deduplicate(

                filtered_items,

                similarity_threshold=0.90
            )

            print(

                f"[Retriever] "

                f"Final unique chunks: "

                f"{len(filtered_items)}"
            )

            # --------------------------------
            # Extract final docs
            # --------------------------------
            final_documents = [

                item["document"]

                for item in filtered_items
            ]

            final_distances = [

                item["distance"]

                for item in filtered_items
            ]

            # --------------------------------
            # Return final top_k
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