import ollama


class LLMEngine:

    def __init__(

        self,

        model="iKhalid/ALLaM:7b"
    ):

        self.model = model

    # -----------------------------------
    # Dynamic response length
    # -----------------------------------
    def get_max_tokens(

        self,

        intent
    ):

        token_map = {

            # --------------------------------
            # Direct factual responses
            # --------------------------------
            "market_data": 140,

            # --------------------------------
            # Forecast reasoning
            # --------------------------------
            "forecast": 220,

            # --------------------------------
            # Deeper analytical answers
            # --------------------------------
            "analysis": 320,

            # --------------------------------
            # Educational finance
            # --------------------------------
            "general_finance": 260
        }

        return token_map.get(
            intent,
            220
        )

    # -----------------------------------
    # Main Generation Pipeline
    # -----------------------------------
    def generate(

        self,

        question,

        context=None,

        intent="general_finance"
    ):

        try:

            # -----------------------------------
            # Dynamic token control
            # -----------------------------------
            max_tokens = self.get_max_tokens(
                intent
            )

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
- Keep the response concise but useful
- Focus on the most important insights
- Avoid unnecessary repetition
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
- Keep the answer concise
- Focus on useful information
- Avoid unnecessary details
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
                ],

                options={

                    "temperature": 0.2,

                    "num_predict": max_tokens
                }
            )

            return response["message"]["content"]

        except Exception as e:

            return (
                f"Error generating response: "
                f"{str(e)}"
            )