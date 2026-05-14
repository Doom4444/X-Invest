import ollama


class LLMEngine:

    def __init__(self, model="iKhalid/ALLaM:7b"):

        self.model = model

    # -----------------------------------
    # Main Generation Pipeline
    # -----------------------------------
    def generate(self, question, context=None):

        try:

            # -----------------------------------
            # RAG MODE
            # -----------------------------------
            if context and context.strip():

                prompt = f"""
You are a financial AI assistant.

Use the provided context as the primary source of truth.

Instructions:
- Answer clearly and professionally
- Keep explanations simple for beginners
- If financial numbers exist in the context:
  preserve them exactly
- Do not change percentages, revenues,
  prices, or dates
- Do not invent facts not supported
  by the context
- If the context partially answers
  the question:
  combine the context with careful
  general financial knowledge
- If the answer is completely missing:
  clearly say the information
  is not available

Retrieved Context:
{context}

Question:
{question}

Answer:
"""

            # -----------------------------------
            # KNOWLEDGE MODE
            # -----------------------------------
            else:

                prompt = f"""
You are a financial education assistant.

Answer using general financial knowledge.

Rules:
- Explain clearly and simply
- Answer directly
- Do NOT invent financial numbers
- Do NOT give financial advice
- Use financial terms only if needed
- If unsure, avoid unsupported claims

Question:
{question}

Answer:
"""

            # -----------------------------------
            # Generate Response
            # -----------------------------------
            response = ollama.chat(

                model=self.model,

                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            return response["message"]["content"]

        except Exception as e:

            return (
                f"Error generating response: "
                f"{str(e)}"
            )