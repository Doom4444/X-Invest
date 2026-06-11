from transformers import pipeline


class NewsAnalyzer:

    def __init__(self):

        print(
            "[NewsAnalyzer] "
            "Loading FinBERT..."
        )

        self.classifier = pipeline(

            "text-classification",

            model="ProsusAI/finbert"
        )

    # -----------------------------------
    # Analyze sentiment
    # -----------------------------------
    def analyze(self, articles):

        if not articles:

            return {

                "sentiment": "neutral",

                "confidence": 0,

                "summary": "No news available."
            }

        headlines = []

        for article in articles:

            headline = article.get(
                "headline",
                ""
            )

            if headline:

                headlines.append(
                    headline
                )

        if not headlines:

            return {

                "sentiment": "neutral",

                "confidence": 0,

                "summary": "No headlines available."
            }

        # -----------------------------
        # Run sentiment analysis
        # -----------------------------
        results = self.classifier(
            headlines
        )

        scores = {

            "positive": 0,

            "negative": 0,

            "neutral": 0
        }

        confidences = []

        for result in results:

            label = (
                result["label"]
                .lower()
            )

            score = result["score"]

            if label in scores:

                scores[label] += 1

                confidences.append(
                    score
                )

        # -----------------------------
        # Final sentiment
        # -----------------------------
        final_sentiment = max(

            scores,

            key=scores.get
        )

        avg_confidence = round(

            sum(confidences) /
            len(confidences),

            3
        )

        # -----------------------------
        # Summary
        # -----------------------------
        summary = (

            f"Market sentiment appears "
            f"{final_sentiment} "
            f"based on recent news."
        )

        return {

            "sentiment": final_sentiment,

            "confidence": avg_confidence,

            "summary": summary
        }