import re
import numpy as np
import ollama

from sklearn.metrics.pairwise import cosine_similarity


class QueryAnalyzer:

    def __init__(self):

        # -----------------------------------
        # Models
        # -----------------------------------
        self.embedding_model = (
            "bge-m3"
        )

        self.llm_model = (
            "iKhalid/ALLaM:7b"
        )

        # -----------------------------------
        # Semantic threshold
        # -----------------------------------
        self.semantic_threshold = 0.72

        # -----------------------------------
        # Intent examples
        # -----------------------------------
        self.intents = {

            # --------------------------------
            # Market Data
            # --------------------------------
            "market_data": [

                "tesla live quote",

                "bitcoin current quote",

                "gold trading at",

                "apple ticker",

                "oil live quote"
            ],

            # --------------------------------
            # General Finance
            # --------------------------------
            "general_finance": [

                "what is diversification",

                "what is investing",

                "define portfolio",

                "what is inflation",

                "difference between stocks and bonds",

                "what is cpi",

                "what is monetary policy",

                "what are corporate bonds"
            ],

            # --------------------------------
            # Analysis
            # --------------------------------
            "analysis": [

                "financial report analysis",

                "economic outlook",

                "company performance",

                "financial statement analysis",

                "macroeconomic analysis",

                "market analysis",

                "global economy outlook",

                "economic growth risks"
            ],

            # --------------------------------
            # Forecast
            # --------------------------------
            "forecast": [

                "predict tesla future trend",

                "future outlook for bitcoin",

                "forecast stock movement",

                "next year market forecast"
            ],

            # --------------------------------
            # News Analysis
            # --------------------------------
            "news_analysis": [

                "latest bitcoin news",

                "breaking market news",

                "why is tesla falling today",

                "recent economic news",

                "market headlines"
            ]
        }

        # -----------------------------------
        # Precompute embeddings
        # -----------------------------------
        self.intent_embeddings = {}

        self.prepare_intent_embeddings()

    # -----------------------------------
    # Prepare embeddings once
    # -----------------------------------
    def prepare_intent_embeddings(self):

        for intent, examples in (
            self.intents.items()
        ):

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

    # -----------------------------------
    # Rule Layer
    # -----------------------------------
    def rule_layer(self, question):

        q = question.lower()

        # -----------------------------------
        # Forecast FIRST
        # -----------------------------------
        if re.search(

            r"predict|forecast|future outlook|next year",

            q
        ):

            return (
                "forecast",
                0.95,
                "rule"
            )

        # -----------------------------------
        # News Analysis
        # -----------------------------------
        if re.search(

            r"latest news|breaking news|headline|falling today|dropping today|recent news",

            q
        ):

            return (
                "news_analysis",
                0.92,
                "rule"
            )

        # -----------------------------------
        # Market Data
        # -----------------------------------
        if re.search(

            r"ticker|live quote|current quote|trading at",

            q
        ):

            return (
                "market_data",
                0.90,
                "rule"
            )

        # -----------------------------------
        # Analysis
        # -----------------------------------
        if re.search(

            r"financial report|economic outlook|company performance|market analysis",

            q
        ):

            return (
                "analysis",
                0.90,
                "rule"
            )

        return None, 0, None

    # -----------------------------------
    # Semantic Layer
    # -----------------------------------
    def embedding_layer(self, question):

        q_emb = ollama.embeddings(

            model=self.embedding_model,

            prompt=question

        )["embedding"]

        q_emb = np.array(q_emb).reshape(
            1,
            -1
        )

        best_intent = None

        best_score = 0

        for intent, emb in (
            self.intent_embeddings.items()
        ):

            score = cosine_similarity(

                q_emb,
                emb

            ).max()

            if score > best_score:

                best_score = score

                best_intent = intent

        return best_intent, best_score

    # -----------------------------------
    # LLM Fallback
    # -----------------------------------
    def llm_layer(self, question):

        prompt = f"""
Classify the following financial question
into ONE category.

Categories:
market_data
general_finance
analysis
forecast
news_analysis

Rules:
- market_data:
  live prices and current values

- general_finance:
  educational concepts

- analysis:
  reports and financial discussions

- forecast:
  future predictions

- news_analysis:
  recent events and news

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

            .lower()
        )

    # -----------------------------------
    # Routing Metadata
    # -----------------------------------
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

            "needs_market_data": False,

            "needs_rag": False
        }

        # -----------------------------------
        # Intent Routing
        # -----------------------------------
        if intent == "forecast":

            metadata[
                "needs_prediction"
            ] = True

            metadata[
                "needs_market_data"
            ] = True

        elif intent == "market_data":

            metadata[
                "needs_market_data"
            ] = True

        elif intent == "news_analysis":

            metadata[
                "needs_news"
            ] = True

            metadata[
                "needs_market_data"
            ] = True

        elif intent == "general_finance":

            metadata[
                "needs_rag"
            ] = True

        elif intent == "analysis":

            metadata[
                "needs_rag"
            ] = True

            metadata[
                "needs_news"
            ] = True

        return metadata

    # -----------------------------------
    # Main Analyzer
    # -----------------------------------
    def analyze(self, question):

        # -----------------------------------
        # 1. Rule Layer
        # -----------------------------------
        intent, confidence, method = (

            self.rule_layer(question)
        )

        if confidence > 0.9:

            print(
                "[Analyzer] Rule match"
            )

            return self.build_metadata(

                intent,

                confidence,

                method
            )

        # -----------------------------------
        # 2. Semantic Layer
        # -----------------------------------
        intent, score = (
            self.embedding_layer(question)
        )

        print(

            f"[Analyzer] Semantic score: "

            f"{round(score, 3)}"
        )

        if score > self.semantic_threshold:

            print(
                "[Analyzer] Semantic match"
            )

            return self.build_metadata(

                intent,

                score,

                "semantic"
            )

        # -----------------------------------
        # 3. LLM Fallback
        # -----------------------------------
        print(
            "[Analyzer] Fallback to LLM"
        )

        intent = self.llm_layer(
            question
        )

        return self.build_metadata(

            intent,

            0.65,

            "llm"
        )