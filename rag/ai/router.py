from rag.ai.query_analyzer import QueryAnalyzer

from rag.core.rag_engine import RAGEngine
from rag.ai.llm_engine import LLMEngine

from rag.online.asset_extractor import AssetExtractor
from rag.online.market_fetcher import MarketFetcher
from rag.online.market_verifier import MarketVerifier

from rag.online.news_fetcher import NewsFetcher
from rag.online.news_analyzer import NewsAnalyzer

from rag.core.context_fusion import ContextFusion


class Router:

    def __init__(self):

        # -----------------------------------
        # Core systems
        # -----------------------------------
        self.analyzer = QueryAnalyzer()

        self.rag = RAGEngine()

        self.llm = LLMEngine()

        # -----------------------------------
        # Fusion layer
        # -----------------------------------
        self.fusion = ContextFusion()

        # -----------------------------------
        # Market systems
        # -----------------------------------
        self.asset_extractor = AssetExtractor()

        self.market_fetcher = MarketFetcher()

        self.market_verifier = MarketVerifier()

        # -----------------------------------
        # News systems
        # -----------------------------------
        self.news_fetcher = NewsFetcher()

        self.news_analyzer = NewsAnalyzer()

        # -----------------------------------
        # Forecast placeholder
        # -----------------------------------
        self.forecast_model = None

    # -----------------------------------
    # Confidence calculation
    # -----------------------------------
    def calculate_confidence(self, distances):

        if not distances:

            return 70.0

        similarities = []

        for d in distances:

            similarity = (
                1 / (1 + abs(d))
            )

            similarities.append(
                similarity
            )

        avg_similarity = (

            sum(similarities) /
            len(similarities)
        )

        strong_matches = [

            s for s in similarities

            if s > 0.75
        ]

        consistency = (

            len(strong_matches) /
            len(similarities)
        )

        confidence = round(

            (
                (0.6 * avg_similarity) +
                (0.4 * consistency)
            ) * 100,

            1
        )

        return confidence

    # -----------------------------------
    # Main routing pipeline
    # -----------------------------------
    def route(self, question):

        # -----------------------------------
        # 1. Analyze question
        # -----------------------------------
        analysis = self.analyzer.analyze(
            question
        )

        print("\n[Router] Analysis:")

        print(analysis)

        # -----------------------------------
        # 2. Shared extraction
        # -----------------------------------
        asset = None

        symbol = None

        if (

            analysis["needs_market_data"]

            or analysis["needs_news"]

            or analysis["needs_prediction"]

        ):

            asset = self.asset_extractor.extract(
                question
            )

            symbol = asset.get(
                "possible_symbol"
            )

        # -----------------------------------
        # 3. Context containers
        # -----------------------------------
        rag_context = ""

        market_context = ""

        news_context = ""

        prediction_context = ""

        confidence_scores = []

        # -----------------------------------
        # 4. Conditional RAG
        # -----------------------------------
        if analysis["needs_rag"]:

            print(
                "[Router] Running RAG..."
            )

            rag_response = self.rag.ask(
                question
            )

            rag_context = (
                rag_response.get(
                    "context",
                    ""
                )
            )

            if rag_context.strip():

                print(
                    "[Router] "
                    "RAG context added"
                )

                rag_confidence = (

                    self.calculate_confidence(
                        rag_response.get(
                            "distances"
                        )
                    )
                )

                confidence_scores.append(
                    rag_confidence
                )

            else:

                print(
                    "[Router] "
                    "No useful RAG context"
                )

        # -----------------------------------
        # 5. Market Data Layer
        # -----------------------------------
        if analysis["needs_market_data"]:

            print(
                "[Router] "
                "Fetching market data..."
            )

            if symbol:

                results = (

                    self.market_fetcher
                    .fetch_prices(symbol)
                )

                if results:

                    price, market_confidence = (

                        self.market_verifier
                        .verify(results)
                    )

                    market_context = (

                        f"{symbol} current "
                        f"price is ${price}"
                    )

                    if market_confidence is not None:

                        confidence_scores.append(
                            market_confidence
                        )

                else:

                    print(
                        "[Router] "
                        "Market fetch failed"
                    )

            else:

                print(
                    "[Router] "
                    "No symbol detected"
                )

        # -----------------------------------
        # 6. Forecast Layer
        # -----------------------------------
        if analysis["needs_prediction"]:

            print(
                "[Router] "
                "Running prediction..."
            )

            if self.forecast_model:

                prediction, prob = (

                    self.forecast_model
                    .predict(question)
                )

                prediction_context = (
                    prediction
                )

                confidence_scores.append(
                    prob * 100
                )

            else:

                prediction_context = (

                    "Forecast model "
                    "not available yet."
                )

        # -----------------------------------
        # 7. News Layer
        # -----------------------------------
        if analysis["needs_news"]:

            print(
                "[Router] "
                "Running news analysis..."
            )

            if symbol:

                # -----------------------------
                # Fetch articles
                # -----------------------------
                articles = (

                    self.news_fetcher
                    .fetch_news(symbol)
                )

                # -----------------------------
                # Analyze sentiment
                # -----------------------------
                sentiment_data = (

                    self.news_analyzer
                    .analyze(articles)
                )

                # -----------------------------
                # Build headlines context
                # -----------------------------
                headlines_context = (

                    self.news_fetcher
                    .build_news_context(
                        articles
                    )
                )

                # -----------------------------
                # Build final news context
                # -----------------------------
                news_context = (

                    f"{sentiment_data['summary']}\n\n"

                    f"Detected sentiment: "
                    f"{sentiment_data['sentiment']}\n"

                    f"Sentiment confidence: "
                    f"{round(sentiment_data['confidence'] * 100, 1)}%\n\n"

                    f"{headlines_context}"
                )

                # -----------------------------
                # Add confidence
                # -----------------------------
                confidence_scores.append(

                    sentiment_data[
                        "confidence"
                    ] * 100
                )

            else:

                print(
                    "[Router] "
                    "No symbol for news fetch"
                )

        # -----------------------------------
        # 8. Context Fusion
        # -----------------------------------
        fused_context = self.fusion.fuse(

            rag_context=rag_context,

            market_context=market_context,

            news_context=news_context,

            prediction_context=prediction_context
        )

        print("\n[Router] Fused Context:")

        print(fused_context)

        # -----------------------------------
        # 9. Final Response Generation
        # -----------------------------------
        if fused_context.strip():

            final_answer = self.llm.generate(

                question=question,

                context=fused_context
            )

        else:

            final_answer = self.llm.generate(
                question=question
            )

        # -----------------------------------
        # 10. Final Confidence
        # -----------------------------------
        if confidence_scores:

            final_confidence = round(

                sum(confidence_scores) /
                len(confidence_scores),

                1
            )

        else:

            final_confidence = 65.0

        # -----------------------------------
        # 11. Return response
        # -----------------------------------
        return {

            "question": question,

            "intent": analysis["intent"],

            "answer": final_answer,

            "confidence": f"{final_confidence}%",

            "analysis": analysis,

            "context": fused_context
        }