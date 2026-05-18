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
        self.semantic_threshold = 0.68

        # -----------------------------------
        # Intent Definitions
        # -----------------------------------
        self.intent_definitions = {

            # --------------------------------
            # Market Data
            # --------------------------------
            "market_data": {

                "description":
                    (
                        "Questions asking for live financial "
                        "asset prices, current stock values, "
                        "real-time cryptocurrency prices, "
                        "market quotes, trading values, "
                        "or current market valuation data"
                    ),

                "examples": [

                    "tesla stock price today",

                    "bitcoin current value",

                    "gold live quote"
                ]
            },

            # --------------------------------
            # General Finance
            # --------------------------------
            "general_finance": {

                "description":
                    (
                        "Educational finance questions asking "
                        "for explanations of financial concepts, "
                        "investment terminology, economic definitions, "
                        "banking concepts, portfolio concepts, "
                        "or beginner-friendly financial learning"
                    ),

                "examples": [

                    "what is inflation",

                    "difference between stocks and bonds",

                    "what is diversification"
                ]
            },

            # --------------------------------
            # Analysis
            # --------------------------------
            "analysis": {

                "description":
                    (
                        "Questions about financial reports, "
                        "economic conditions, company performance, "
                        "macroeconomic analysis, monetary policy, "
                        "central bank reports, financial operations, "
                        "risk discussions, financial statements, "
                        "or analytical interpretation of financial data"
                    ),

                "examples": [

                    "what does the report say about inflation",

                    "federal reserve economic outlook",

                    "financial performance analysis"
                ]
            },

            # --------------------------------
            # Forecast
            # --------------------------------
            "forecast": {

                "description":
                    (
                        "Questions asking about future market expectations, "
                        "future stock movement, financial forecasting, "
                        "trend prediction, future asset outlook, "
                        "or expected future market behavior"
                    ),

                "examples": [

                    "predict tesla future price",

                    "future outlook for bitcoin",

                    "forecast stock movement"
                ]
            },

            # --------------------------------
            # News Analysis
            # --------------------------------
            "news_analysis": {

                "description":
                    (
                        "Questions about recent financial news, "
                        "breaking economic events, market headlines, "
                        "recent company news, current market sentiment, "
                        "or discussions about recent financial events "
                        "affecting markets or assets"
                    ),

                "examples": [

                    "latest bitcoin news",

                    "recent market headlines",

                    "why is tesla falling today"
                ]
            }
        }

        # -----------------------------------
        # Store intent embeddings
        # -----------------------------------
        self.intent_embeddings = {}

        # -----------------------------------
        # Build embeddings once
        # -----------------------------------
        self.prepare_intent_embeddings()

    # -----------------------------------
    # Prepare semantic intent embeddings
    # -----------------------------------
    def prepare_intent_embeddings(self):

        for intent, data in (

            self.intent_definitions.items()
        ):

            texts = []

            # --------------------------------
            # Description
            # --------------------------------
            texts.append(
                data["description"]
            )

            # --------------------------------
            # Examples
            # --------------------------------
            texts.extend(
                data["examples"]
            )

            embeddings = []

            for text in texts:

                emb = ollama.embeddings(

                    model=self.embedding_model,

                    prompt=text

                )["embedding"]

                embeddings.append(emb)

            self.intent_embeddings[intent] = (
                np.array(embeddings)
            )

    # -----------------------------------
    # Lightweight Rules
    # -----------------------------------
    def rule_layer(self, question):

        q = question.lower()

        # -----------------------------------
        # Forecast
        # -----------------------------------
        if re.search(

            r"\bpredict\b|\bforecast\b",

            q
        ):

            return (
                "forecast",
                0.95,
                "rule"
            )

        # -----------------------------------
        # Breaking News
        # -----------------------------------
        if re.search(

            r"breaking news|latest headlines",

            q
        ):

            return (
                "news_analysis",
                0.92,
                "rule"
            )

        return None, 0, None

    # -----------------------------------
    # Semantic Intent Detection
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
    # LLM fallback
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

Category meanings:

market_data:
Questions asking for live prices,
market values, or trading information.

general_finance:
Educational financial explanations
and beginner-friendly concepts.

analysis:
Financial reports, economic analysis,
financial discussions, risk analysis,
and company/economic interpretation.

forecast:
Future predictions, trend expectations,
and future market outlooks.

news_analysis:
Recent financial news, headlines,
market sentiment, and current events.

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
    # Build routing metadata
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
        # Routing Logic
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
    # Main Pipeline
    # -----------------------------------
    def analyze(self, question):

        # -----------------------------------
        # 1. Rules
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

        if score >= self.semantic_threshold:

            print(
                "[Analyzer] Semantic match"
            )

            return self.build_metadata(

                intent,

                score,

                "semantic"
            )

        # -----------------------------------
        # 3. LLM fallback
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