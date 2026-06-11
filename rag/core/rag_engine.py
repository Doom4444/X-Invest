from rag.core.retriever import Retriever


class RAGEngine:

    def __init__(self):

        self.retriever = Retriever()

    # -----------------------------
    # Build Context
    # -----------------------------
    def build_context(self, documents):

        return "\n\n".join([
            doc for doc in documents
            if doc
        ])

    # -----------------------------
    # Retrieve + Build Context
    # -----------------------------
    def ask(self, question):

        # 1. Retrieve documents
        documents, distances = (
            self.retriever.retrieve(question)
        )

        # 2. Build context
        context = self.build_context(documents)

        # 3. Detect mode
        mode = "rag" if context.strip() else "knowledge"

        # 4. Return structured response
        return {

            "mode": mode,

            "context": context,

            "documents": documents,

            "distances": distances
        }