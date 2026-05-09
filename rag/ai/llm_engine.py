import ollama


class LLMEngine:

    def __init__(self, model="iKhalid/ALLaM:7b"):
        self.model = model

    def generate(self, question, context=None):

        try:

            # -----------------------------
            # RAG MODE
            # -----------------------------
            if context and context.strip():

                prompt = f"""
You are a financial assistant.

Use ONLY the provided context to answer the question.

Rules:
- Be clear and simple for beginners
- Answer directly, do not repeat the question
- If numbers exist → include them exactly
- Do NOT change financial values
- If answer is not found → say clearly it's not available

Context:
{context}

Question:
{question}

Answer:
"""

            # -----------------------------
            # KNOWLEDGE MODE
            # -----------------------------
            else:

                prompt = f"""
You are a financial education assistant.

Answer using your knowledge.

Rules:
- Explain clearly and simply
- Answer directly
- Do NOT invent financial numbers
- Do NOT give financial advice
- Use financial terms only if needed

Question:
{question}

Answer:
"""

            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}]
            )

            return response["message"]["content"]

        except Exception as e:
            return f"Error generating response: {str(e)}"