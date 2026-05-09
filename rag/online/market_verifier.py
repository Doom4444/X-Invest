import re
import json
import ollama


class AssetExtractor:

    def __init__(self):

        self.model = "iKhalid/ALLaM:7b"

        # -----------------------------
        # Known assets dictionary
        # -----------------------------
        self.known_assets = {

            # Stocks
            "tesla": ("Tesla", "stock", "TSLA"),
            "apple": ("Apple", "stock", "AAPL"),
            "microsoft": ("Microsoft", "stock", "MSFT"),

            # Crypto
            "bitcoin": ("Bitcoin", "crypto", "BTC"),
            "btc": ("Bitcoin", "crypto", "BTC"),
            "ethereum": ("Ethereum", "crypto", "ETH"),

            # Commodities
            "gold": ("Gold", "commodity", "XAUUSD"),
            "silver": ("Silver", "commodity", "XAGUSD"),
            "oil": ("Oil", "commodity", "CL"),

            # Forex
            "eurusd": ("EUR/USD", "forex", "EURUSD"),
            "usd/jpy": ("USD/JPY", "forex", "USDJPY")
        }

    # -----------------------------
    # Rule Layer
    # -----------------------------
    def rule_extract(self, question):

        q = question.lower()

        for key, value in self.known_assets.items():

            if key in q:

                return {
                    "asset_name": value[0],
                    "asset_type": value[1],
                    "possible_symbol": value[2],
                    "method": "rule"
                }

        return None

    # -----------------------------
    # LLM Fallback
    # -----------------------------
    def llm_extract(self, question):

        prompt = f"""
Extract the financial asset from the question.

Return ONLY valid JSON.

Format:
{{
  "asset_name": "...",
  "asset_type": "...",
  "possible_symbol": "..."
}}

Question:
{question}
"""

        try:

            response = ollama.chat(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            text = response["message"]["content"].strip()

            # remove markdown json formatting
            text = re.sub(r"```json|```", "", text).strip()

            data = json.loads(text)

            data["method"] = "llm"

            return data

        except Exception as e:

            print("[AssetExtractor] LLM extraction failed:", e)

            return {
                "asset_name": None,
                "asset_type": None,
                "possible_symbol": None,
                "method": "failed"
            }

    # -----------------------------
    # Main Extraction
    # -----------------------------
    def extract(self, question):

        # 1. Rule-based extraction
        result = self.rule_extract(question)

        if result:

            print("[AssetExtractor] Rule match")

            return result

        # 2. LLM fallback
        print("[AssetExtractor] Using LLM fallback")

        return self.llm_extract(question)