import re
import json
import ollama


class AssetExtractor:

    def __init__(self):

        self.model = "iKhalid/ALLaM:7b"

        # -----------------------------------
        # Known assets dictionary
        # -----------------------------------
        self.known_assets = {

            # --------------------------------
            # Stocks
            # --------------------------------
            "tesla": (
                "Tesla",
                "stock",
                "TSLA"
            ),

            "apple": (
                "Apple",
                "stock",
                "AAPL"
            ),

            "microsoft": (
                "Microsoft",
                "stock",
                "MSFT"
            ),

            "amazon": (
                "Amazon",
                "stock",
                "AMZN"
            ),

            "google": (
                "Google",
                "stock",
                "GOOGL"
            ),

            "meta": (
                "Meta",
                "stock",
                "META"
            ),

            "nvidia": (
                "NVIDIA",
                "stock",
                "NVDA"
            ),

            # --------------------------------
            # Crypto
            # --------------------------------
            "bitcoin": (
                "Bitcoin",
                "crypto",
                "BTC-USD"
            ),

            "btc": (
                "Bitcoin",
                "crypto",
                "BTC-USD"
            ),

            "ethereum": (
                "Ethereum",
                "crypto",
                "ETH-USD"
            ),

            "eth": (
                "Ethereum",
                "crypto",
                "ETH-USD"
            ),

            # --------------------------------
            # Commodities
            # --------------------------------
            "gold": (
                "Gold",
                "commodity",
                "GC=F"
            ),

            "silver": (
                "Silver",
                "commodity",
                "SI=F"
            ),

            "oil": (
                "Oil",
                "commodity",
                "CL=F"
            ),

            # --------------------------------
            # Forex
            # --------------------------------
            "eurusd": (
                "EUR/USD",
                "forex",
                "EUR/USD"
            ),

            "usdjpy": (
                "USD/JPY",
                "forex",
                "USD/JPY"
            ),

            "usd/jpy": (
                "USD/JPY",
                "forex",
                "USD/JPY"
            )
        }

    # -----------------------------------
    # Rule-based Extraction
    # -----------------------------------
    def rule_extract(self, question):

        q = question.lower()

        for key, value in (
            self.known_assets.items()
        ):

            if key in q:

                return {

                    "asset_name": value[0],

                    "asset_type": value[1],

                    "possible_symbol": value[2],

                    "method": "rule"
                }

        return None

    # -----------------------------------
    # Clean JSON Text
    # -----------------------------------
    def clean_json_response(self, text):

        # --------------------------------
        # Remove markdown formatting
        # --------------------------------
        text = re.sub(

            r"```json|```",

            "",

            text
        ).strip()

        # --------------------------------
        # Extract JSON block only
        # --------------------------------
        match = re.search(

            r"\{.*\}",

            text,

            re.DOTALL
        )

        if match:

            return match.group()

        return text

    # -----------------------------------
    # Validate JSON Structure
    # -----------------------------------
    def validate_response(self, data):

        required_keys = [

            "asset_name",

            "asset_type",

            "possible_symbol"
        ]

        if not isinstance(data, dict):

            return False

        for key in required_keys:

            if key not in data:

                return False

        return True

    # -----------------------------------
    # LLM Fallback Extraction
    # -----------------------------------
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

Rules:
- Use real market symbols
- For Bitcoin use BTC-USD
- For Ethereum use ETH-USD
- For Gold use GC=F
- For Oil use CL=F
- If unknown:
  set fields to null

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

            text = (

                response["message"]["content"]

                .strip()
            )

            # --------------------------------
            # Clean model response
            # --------------------------------
            text = self.clean_json_response(
                text
            )

            # --------------------------------
            # Parse JSON
            # --------------------------------
            data = json.loads(text)

            # --------------------------------
            # Validate output
            # --------------------------------
            if not self.validate_response(
                data
            ):

                raise ValueError(
                    "Invalid JSON structure"
                )

            # --------------------------------
            # Add metadata
            # --------------------------------
            data["method"] = "llm"

            return data

        except Exception as e:

            print(

                "[AssetExtractor] "

                f"LLM extraction failed: {e}"
            )

            return {

                "asset_name": None,

                "asset_type": None,

                "possible_symbol": None,

                "method": "failed"
            }

    # -----------------------------------
    # Main Extraction Pipeline
    # -----------------------------------
    def extract(self, question):

        # --------------------------------
        # 1. Rule-based extraction
        # --------------------------------
        result = self.rule_extract(
            question
        )

        if result:

            print(
                "[AssetExtractor] Rule match"
            )

            return result

        # --------------------------------
        # 2. LLM fallback
        # --------------------------------
        print(

            "[AssetExtractor] "

            "Using LLM fallback"
        )

        return self.llm_extract(
            question
        )