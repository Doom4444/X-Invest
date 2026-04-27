import ollama
import json


class AssetExtractor:

    def __init__(self):

        self.model = "iKhalid/ALLaM:7b"

    def extract(self, question):

        prompt = f"""
You are a financial entity extraction system.

Your task is to analyze a user question and identify the financial asset being referenced.

The asset may belong to ANY financial category including but not limited to:
- Stocks
- Cryptocurrencies
- Forex currencies
- Commodities
- Market indices
- ETFs
- Bonds

Instructions:
1. Identify the asset mentioned in the question.
2. Determine the most likely asset type.
3. Extract the most relevant symbol or ticker if possible.
4. If no clear ticker exists, return the asset name only.

Return your answer ONLY as JSON in the following format:

{{
 "asset_name": "...",
 "asset_type": "...",
 "possible_symbol": "..."
}}

Examples:

Question: What is Tesla stock price?
{{
 "asset_name": "Tesla",
 "asset_type": "stock",
 "possible_symbol": "TSLA"
}}

Question: What is gold price today?
{{
 "asset_name": "Gold",
 "asset_type": "commodity",
 "possible_symbol": "XAUUSD"
}}

Question: Bitcoin price now
{{
 "asset_name": "Bitcoin",
 "asset_type": "crypto",
 "possible_symbol": "BTC"
}}

Question: USD to EUR rate
{{
 "asset_name": "USD/EUR",
 "asset_type": "forex",
 "possible_symbol": "EURUSD"
}}

Now analyze this question:

Question:
{question}
"""

        try:

            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}]
            )

            text = response["message"]["content"].strip()

            data = json.loads(text)

            return data

        except Exception:

            
            return {
                "asset_name": None,
                "asset_type": None,
                "possible_symbol": None
            }