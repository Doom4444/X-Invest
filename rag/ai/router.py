from rag.ai.query_analyzer import QueryAnalyzer

from rag.core.rag_engine import RAGEngine
from rag.ai.llm_engine import LLMEngine

from rag.online.asset_extractor import AssetExtractor
from rag.online.market_fetcher import MarketFetcher
from rag.online.market_verifier import MarketVerifier


class Router:

    def __init__(self):

        # Core systems
        self.analyzer = QueryAnalyzer()
        self.rag = RAGEngine()
        self.llm = LLMEngine()

        # Online market layer
        self.asset_extractor = AssetExtractor()
        self.market_fetcher = MarketFetcher()
        self.market_verifier = MarketVerifier()

        # Forecast placeholder
        self.forecast_model = None

        # News placeholder
        self.news_system = None

    # -----------------------------------
    # Confidence calculation
    # -----------------------------------
    def calculate_confidence(self, distances):

        if not distances:
            return 70.0

        similarities = [1 - d for d in distances]

        avg_similarity = sum(similarities) / len(similarities)

        strong_matches = [
            s for s in similarities
            if s > 0.75
        ]

        consistency = len(strong_matches) / len(similarities)

        confidence = round(
            ((0.6 * avg_similarity) +
             (0.4 * consistency)) * 100,
            1
        )

        return confidence

    # -----------------------------------
    # Main routing pipeline
    # -----------------------------------
    def route(self, question):

        # -----------------------------
        # 1. Analyze question
        # -----------------------------
        analysis = self.analyzer.analyze(question)

        print("\n[Router] Analysis:")
        print(analysis)

        # -----------------------------
        # 2. Context containers
        # -----------------------------
        contexts = []

        confidence_scores = []

        # -----------------------------
        # 3. RAG Layer
        # -----------------------------
        if analysis["needs_rag"]:

            print("[Router] Running RAG...")

            rag_response = self.rag.ask(question)

            contexts.append(
                f"RAG INFORMATION:\n{rag_response['answer']}"
            )

            rag_confidence = self.calculate_confidence(
                rag_response.get("distances")
            )

            confidence_scores.append(rag_confidence)

        # -----------------------------
        # 4. Market Data Layer
        # -----------------------------
        if analysis["needs_market_data"]:

            print("[Router] Fetching market data...")

            asset = self.asset_extractor.extract(question)

            symbol = asset.get("possible_symbol")

            if symbol:

                results = self.market_fetcher.fetch_prices(symbol)

                if results:

                    price, market_confidence = (
                        self.market_verifier.verify(results)
                    )

                    contexts.append(
                        f"MARKET DATA:\n"
                        f"{symbol} current price is ${price}"
                    )

                    confidence_scores.append(
                        float(str(market_confidence).replace("%", ""))
                    )

        # -----------------------------
        # 5. Forecast Layer
        # -----------------------------
        if analysis["needs_prediction"]:

            print("[Router] Running prediction...")

            if self.forecast_model:

                prediction, prob = (
                    self.forecast_model.predict(question)
                )

                contexts.append(
                    f"PREDICTION:\n{prediction}"
                )

                confidence_scores.append(prob * 100)

            else:

                contexts.append(
                    "PREDICTION:\nForecast model not available yet."
                )

        # -----------------------------
        # 6. News Layer
        # -----------------------------
        if analysis["needs_news"]:

            print("[Router] Running news analysis...")

            # Placeholder for future News AI
            contexts.append(
                "NEWS ANALYSIS:\n"
                "News analysis system will be integrated soon."
            )

        # -----------------------------
        # 7. Context Fusion
        # -----------------------------
        fused_context = "\n\n".join(contexts)

        # -----------------------------
        # 8. Final Response Generation
        # -----------------------------
        final_answer = self.llm.generate(
            question=question,
            context=fused_context
        )

        # -----------------------------
        # 9. Final Confidence
        # -----------------------------
        if confidence_scores:

            final_confidence = round(
                sum(confidence_scores) /
                len(confidence_scores),
                1
            )

        else:

            final_confidence = 65.0

        # -----------------------------
        # 10. Return response
        # -----------------------------
        return {

            "question": question,

            "intent": analysis["intent"],

            "answer": final_answer,

            "confidence": final_confidence,

            "analysis": analysis,

            "context": fused_context
        }