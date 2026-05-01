from rag.retriever import Retriever
from rag.llm_engine import LLMEngine


class RAGEngine:

    def __init__(self):
        self.retriever = Retriever()
        self.llm = LLMEngine()

    # -----------------------------
    # Build Context
    # -----------------------------
    def build_context(self, documents):
        return "\n\n".join([doc for doc in documents if doc])

    # -----------------------------
    # Ask (Main Pipeline)
    # -----------------------------
    def ask(self, question):

        # 1. Retrieve
        documents, distances = self.retriever.retrieve(question)

        # 2. Build Context
        context = self.build_context(documents)

        # 3. Decide Mode (Smart Routing)
        if not context.strip():
            # 🔥 مفيش داتا → Knowledge Mode
            answer = self.llm.generate(question, context=None)
            mode = "knowledge"
        else:
            # 🔥 فيه داتا → RAG Mode
            answer = self.llm.generate(question, context=context)
            mode = "rag"

        return {
            "question": question,
            "answer": answer,
            "mode": mode,
            "context": context,
            "documents": documents,
            "distances": distances
        }