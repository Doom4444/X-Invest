import ollama


class LLMEngine:

    def __init__(self, model="iKhalid/ALLaM:7b"):
        self.model = model

    def generate(self, question, context=None):

        # لو فيه context → RAG mode
        if context:

            prompt = f"""
You are a financial assistant.

Use ONLY the provided context to answer the question.

Context:
{context}

Question:
{question}

If the answer is not in the context, say:
"The information is not available in the provided documents."
"""

        # لو مفيش context → Knowledge mode
        else:

            prompt = f"""
You are a financial education assistant.

Answer the question using your financial knowledge.

Rules:
- Do NOT invent financial numbers.
- Do NOT give financial advice.
- Provide clear educational explanations.

Question:
{question}
"""

        response = ollama.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )

        return response["message"]["content"]