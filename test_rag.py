from rag.rag_engine import RAGEngine

rag = RAGEngine()

question = "What was the total revenue of WIPO in 2020?"

res = rag.ask(question)

print("\nMode:", res["mode"])
print("\nAnswer:\n")
print(res["answer"])