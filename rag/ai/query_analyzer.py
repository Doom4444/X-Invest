import re
import numpy as np
import ollama

from sklearn.metrics.pairwise import cosine_similarity


class QueryAnalyzer:

    def __init__(self):

        # -----------------------------
        # Models
        # -----------------------------
        self.embedding_model = "nomic-embed-text"

        self.llm_model = "iKhalid/ALLaM:7b"

        # -----------------------------
        # Semantic threshold
        # -----------------------------
        self.semantic_threshold = 0.75

        # -----------------------------
        # Intent examples
        # -----------------------------
        self.intents = {

            "market_data": [

                "tesla stock price today",

                "bitcoin price now",

                "gold current price",

                "latest oil price"
            ],

            "general_finance": [

                "what is diversification",

                "what is investing",

                "define portfolio",

                "what is inflation"
            ],

            "analysis": [

                "should i invest in tesla",

                "portfolio strategy advice",

                "market analysis",

                "stock market risks"
            ],

            "forecast": [

                "predict tesla price",

                "future stock price",

                "market forecast",

                "next year stock prediction"
            ],

            "news_analysis": [

                "why is tesla dropping",

                "latest market news",

                "how news affects stocks",

                "economic news impact"
            ]
        }

        # -----------------------------
        # Precompute intent embeddings
        # -----------------------------
        self.intent_embeddings = {}

        self.prepare_intent_embeddings()

    # -----------------------------
    # Prepare embeddings once
    # -----------------------------
    def prepare_intent_embeddings(self):

        for intent, examples in self.intents.items():

            embeddings = []

            for text in examples:

                emb = ollama.embeddings(

                    model=self.embedding_model,

                    prompt=text
                )["embedding"]

                embeddings.append(emb)

            self.intent_embeddings[intent] = (
                np.array(embeddings)
            )

    # -----------------------------
    # Rule Layer
    # -----------------------------
    def rule_layer(self, question):

        q = question.lower()

        # -----------------------------
        # Forecast FIRST
        # -----------------------------
        if re.search(

            r"predict|forecast|future|next year",

            q
        ):

            return "forecast", 0.95, "rule"

        # -----------------------------
        # Market Data
        # -----------------------------
        if re.search(

            r"price|today|current|now|live|latest|market|trading|ticker|usd|btc|gold|oil",

            q
        ):

            return "market_data", 0.95, "rule"

        # -----------------------------
        # News Analysis
        # -----------------------------
        if re.search(

            r"news|breaking|headline|impact|geopolitical",

            q
        ):

            return "news_analysis", 0.92, "rule"

        return None, 0, None

    # -----------------------------
    # Semantic Layer
    # -----------------------------
    def embedding_layer(self, question):

        q_emb = ollama.embeddings(

            model=self.embedding_model,

            prompt=question

        )["embedding"]

        q_emb = np.array(q_emb).reshape(1, -1)

        best_intent = None

        best_score = 0

        for intent, emb in self.intent_embeddings.items():

            score = cosine_similarity(
                q_emb,
                emb
            ).max()

            if score > best_score:

                best_score = score

                best_intent = intent

        return best_intent, best_score

    # -----------------------------
    # LLM Fallback
    # -----------------------------
    def llm_layer(self, question):

        prompt = f"""
Classify the following finance question
into ONE category.

Categories:
market_data
general_finance
analysis
forecast
news_analysis

Return ONLY the category name.

Question:
{question}
"""

        response = ollama.chat(

            model=self.llm_model,

            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return (
            response["message"]["content"]
            .strip()
        )

    # -----------------------------
    # Routing Metadata
    # -----------------------------
    def build_metadata(
        self,
        intent,
        confidence,
        method
    ):

        metadata = {

            "intent": intent,

            "confidence": round(
                float(confidence),
                3
            ),

            "method": method,

            "needs_prediction": False,

            "needs_news": False,

            "needs_market_data": False
        }

        # -----------------------------
        # Intent Routing
        # -----------------------------
        if intent == "forecast":

            metadata["needs_prediction"] = True

            metadata["needs_market_data"] = True

        elif intent == "market_data":

            metadata["needs_market_data"] = True

        elif intent == "news_analysis":

            metadata["needs_news"] = True

            metadata["needs_market_data"] = True

        elif intent == "analysis":

            metadata["needs_prediction"] = True

            metadata["needs_news"] = True

        return metadata

    # -----------------------------
    # Main Analyzer
    # -----------------------------
    def analyze(self, question):

        # -----------------------------
        # 1. Rule Layer
        # -----------------------------
        intent, confidence, method = (
            self.rule_layer(question)
        )

        if confidence > 0.9:

            print("[Analyzer] Rule match")

            return self.build_metadata(

                intent,

                confidence,

                method
            )

        # -----------------------------
        # 2. Semantic Layer
        # -----------------------------
        intent, score = (
            self.embedding_layer(question)
        )

        print(
            f"[Analyzer] Semantic score: "
            f"{round(score, 3)}"
        )

        if score > self.semantic_threshold:

            print("[Analyzer] Semantic match")

            return self.build_metadata(

                intent,

                score,

                "semantic"
            )

        # -----------------------------
        # 3. LLM Fallback
        # -----------------------------
        print("[Analyzer] Fallback to LLM")

        intent = self.llm_layer(question)

        return self.build_metadata(

            intent,

            0.65,

            "llm"
        )