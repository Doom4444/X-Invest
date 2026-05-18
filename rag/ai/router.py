from rag.ai.query_analyzer import QueryAnalyzer

from rag.core.rag_engine import RAGEngine
from rag.ai.llm_engine import LLMEngine

from rag.online.asset_extractor import AssetExtractor
from rag.online.market_fetcher import MarketFetcher
from rag.online.market_verifier import MarketVerifier

from rag.online.news_fetcher import NewsFetcher
from rag.online.news_analyzer import NewsAnalyzer

from rag.core.context_fusion import ContextFusion
from rag.core.forecast_engine import ForecastEngine

from rag.preprocessing.utils import extract_date


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
        # Forecast system
        # -----------------------------------
        self.forecast_engine = ForecastEngine()

    # -----------------------------------
    # RAG confidence calculation
    # -----------------------------------
    def calculate_confidence(

        self,

        distances,

        analyzer_confidence=0.65
    ):

        # -----------------------------------
        # No retrieval
        # -----------------------------------
        if not distances:

            return round(
                analyzer_confidence * 100,
                1
            )

        similarities = []

        for d in distances:

            # --------------------------------
            # Cosine similarity
            # --------------------------------
            similarity = max(
                0,
                1 - d
            )

            similarities.append(
                similarity
            )

        # -----------------------------------
        # Average similarity
        # -----------------------------------
        avg_similarity = (

            sum(similarities) /
            len(similarities)
        )

        # -----------------------------------
        # Strong retrieval consistency
        # -----------------------------------
        strong_matches = [

            s for s in similarities

            if s >= 0.70
        ]

        consistency = (

            len(strong_matches) /
            len(similarities)
        )

        # -----------------------------------
        # Retrieval confidence
        # -----------------------------------
        retrieval_confidence = (

            (0.7 * avg_similarity) +

            (0.3 * consistency)
        )

        # -----------------------------------
        # Final fusion
        # -----------------------------------
        final_confidence = (

            (0.35 * analyzer_confidence) +

            (0.65 * retrieval_confidence)
        )

        return round(
            final_confidence * 100,
            1
        )

    # -----------------------------------
    # Weighted confidence fusion
    # -----------------------------------
    def fuse_confidence(

        self,

        analyzer_confidence=None,

        rag_confidence=None,

        market_confidence=None,

        news_confidence=None,

        prediction_confidence=None
    ):

        weighted_scores = []

        total_weight = 0

        # -----------------------------------
        # RAG confidence
        # -----------------------------------
        if rag_confidence is not None:

            weighted_scores.append(
                rag_confidence * 0.50
            )

            total_weight += 0.50

        # -----------------------------------
        # Analyzer confidence
        # -----------------------------------
        if analyzer_confidence is not None:

            weighted_scores.append(
                analyzer_confidence * 0.20
            )

            total_weight += 0.20

        # -----------------------------------
        # Market confidence
        # -----------------------------------
        if market_confidence is not None:

            weighted_scores.append(
                market_confidence * 0.20
            )

            total_weight += 0.20

        # -----------------------------------
        # News confidence
        # -----------------------------------
        if news_confidence is not None:

            weighted_scores.append(
                news_confidence * 0.10
            )

            total_weight += 0.10

        # -----------------------------------
        # Prediction confidence
        # -----------------------------------
        if prediction_confidence is not None:

            weighted_scores.append(
                prediction_confidence * 0.15
            )

            total_weight += 0.15

        # -----------------------------------
        # No confidence available
        # -----------------------------------
        if total_weight == 0:

            return 65.0

        # -----------------------------------
        # Final normalized confidence
        # -----------------------------------
        final_confidence = (

            sum(weighted_scores) /
            total_weight
        )

        return round(
            final_confidence,
            1
        )

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
        # 2. Extract target date
        # -----------------------------------
        target_date = extract_date(
            question
        )

        if target_date:

            print(

                "[Router] "

                f"Detected date: "
                f"{target_date}"
            )

        # -----------------------------------
        # 3. Shared extraction
        # -----------------------------------
        asset = {}

        symbol = None

        if (

            analysis["needs_market_data"]

            or analysis["needs_news"]

            or analysis["needs_prediction"]

        ):

            asset = self.asset_extractor.extract(
                question
            )

            if asset:

                symbol = asset.get(
                    "possible_symbol"
                )

        # -----------------------------------
        # 4. Context containers
        # -----------------------------------
        rag_context = ""

        market_context = ""

        news_context = ""

        prediction_context = ""

        # -----------------------------------
        # Confidence containers
        # -----------------------------------
        rag_confidence = None

        market_confidence = None

        news_confidence = None

        prediction_confidence = None

        # -----------------------------------
        # 5. Conditional RAG
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
                        ),

                        analysis["confidence"]
                    )
                )

                print(

                    f"[Router] "

                    f"RAG confidence: "

                    f"{rag_confidence}%"
                )

            else:

                print(
                    "[Router] "
                    "No useful RAG context"
                )

        # -----------------------------------
        # 6. Market Data Layer
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
        # 7. Forecast Layer
        # -----------------------------------
        if analysis["needs_prediction"]:

            print(
                "[Router] "
                "Running forecast..."
            )

            if symbol:

                prediction = (

                    self.forecast_engine
                    .predict(

                        symbol,

                        target_date
                    )
                )

                if prediction:

                    prediction_context = (

                        self.forecast_engine
                        .build_forecast_context(
                            prediction
                        )
                    )

                    prediction_confidence = float(

                        prediction.get(
                            "confidence",
                            70
                        )
                    )

                    print(
                        "[Router] "
                        "Forecast added"
                    )

                else:

                    prediction_context = (

                        "No forecast data "
                        "available for "
                        "this asset."
                    )

                    print(
                        "[Router] "
                        "Forecast failed"
                    )

            else:

                print(
                    "[Router] "
                    "No symbol detected "
                    "for forecast"
                )

        # -----------------------------------
        # 8. News Layer
        # -----------------------------------
        if analysis["needs_news"]:

            print(
                "[Router] "
                "Running news analysis..."
            )

            if symbol:

                articles = (

                    self.news_fetcher
                    .fetch_news(

                        symbol=symbol,

                        asset_name=asset.get(
                            "asset_name"
                        )
                    )
                )

                sentiment_data = (

                    self.news_analyzer
                    .analyze(articles)
                )

                headlines_context = (

                    self.news_fetcher
                    .build_news_context(
                        articles
                    )
                )

                news_context = (

                    f"{sentiment_data['summary']}\n\n"

                    f"Detected sentiment: "
                    f"{sentiment_data['sentiment']}\n"

                    f"Sentiment confidence: "
                    f"{round(sentiment_data['confidence'] * 100, 1)}%\n\n"

                    f"{headlines_context}"
                )

                news_confidence = (

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
        # 9. Context Fusion
        # -----------------------------------
        fused_context = self.fusion.fuse(

            intent=analysis["intent"],

            rag_context=rag_context,

            market_context=market_context,

            news_context=news_context,

            prediction_context=prediction_context
        )

        print("\n[Router] Fused Context:")

        print(fused_context)

        # -----------------------------------
        # 10. Final Response Generation
        # -----------------------------------
        if fused_context.strip():

            final_answer = self.llm.generate(

                question=question,

                context=fused_context,

                intent=analysis["intent"]
            )

        else:

            final_answer = self.llm.generate(

                question=question,

                intent=analysis["intent"]
            )

        # -----------------------------------
        # 11. Final Confidence
        # -----------------------------------
        final_confidence = (

            self.fuse_confidence(

                analyzer_confidence=(

                    analysis["confidence"]
                    * 100
                ),

                rag_confidence=(
                    rag_confidence
                ),

                market_confidence=(
                    market_confidence
                ),

                news_confidence=(
                    news_confidence
                ),

                prediction_confidence=(

                    prediction_confidence
                )
            )
        )

        # -----------------------------------
        # 12. Return response
        # -----------------------------------
        return {

            "question": question,

            "intent": analysis["intent"],

            "answer": final_answer,

            "confidence": f"{final_confidence}%",

            "analysis": analysis,

            "context": fused_context
        }