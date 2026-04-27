from rag.retrieval.retriever import Retriever
from rag.llm.llm_engine import LLMEngine
from rag.utils.numeric_guard import extract_numbers


class RAGEngine:

    def __init__(self):
        self.retriever = Retriever()
        self.llm = LLMEngine()

    def ask(self, question):

        docs = self.retriever.search(question)

        context = "\n\n".join(docs)

        answer = self.llm.generate(question, context)

        numbers = extract_numbers(answer)

        return {
            "answer": answer,
            "numbers_found": numbers
        }