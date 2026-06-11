import requests
import os

from datetime import datetime, timedelta
from dotenv import load_dotenv


load_dotenv()


class NewsFetcher:

    def __init__(self):

        self.api_key = os.getenv(
            "FINNHUB_API_KEY"
        )

        self.base_url = (
            "https://finnhub.io/api/v1/company-news"
        )

        self.timeout = 5

    # -----------------------------------
    # Filter relevant articles
    # -----------------------------------
    def filter_articles(

        self,

        articles,

        symbol,

        asset_name=None
    ):

        filtered = []

        # --------------------------------
        # Normalize entity names
        # --------------------------------
        symbol = (
            symbol.lower()
            if symbol
            else ""
        )

        asset_name = (
            asset_name.lower()
            if asset_name
            else ""
        )

        for article in articles:

            headline = article.get(
                "headline",
                ""
            )

            summary = article.get(
                "summary",
                ""
            )

            text = (

                f"{headline} {summary}"

                .lower()
            )

            # --------------------------------
            # Entity relevance check
            # --------------------------------
            symbol_match = (
                symbol in text
            )

            asset_match = (
                asset_name in text
            )

            # --------------------------------
            # Keep only relevant articles
            # --------------------------------
            if (

                symbol_match

                or asset_match
            ):

                filtered.append(
                    article
                )

        return filtered

    # -----------------------------------
    # Fetch recent news
    # -----------------------------------
    def fetch_news(

        self,

        symbol,

        asset_name=None,

        days_back=7,

        max_articles=5
    ):

        try:

            # -----------------------------
            # Date range
            # -----------------------------
            today = datetime.utcnow()

            past = today - timedelta(
                days=days_back
            )

            to_date = today.strftime(
                "%Y-%m-%d"
            )

            from_date = past.strftime(
                "%Y-%m-%d"
            )

            # -----------------------------
            # API Request
            # -----------------------------
            response = requests.get(

                self.base_url,

                params={

                    "symbol": symbol,

                    "from": from_date,

                    "to": to_date,

                    "token": self.api_key
                },

                timeout=self.timeout
            )

            # -----------------------------
            # Validate
            # -----------------------------
            if response.status_code != 200:

                print(
                    "[NewsFetcher] "
                    f"HTTP {response.status_code}"
                )

                return []

            data = response.json()

            if not isinstance(data, list):

                return []

            # -----------------------------
            # Clean raw articles
            # -----------------------------
            raw_articles = []

            for item in data:

                headline = item.get(
                    "headline",
                    ""
                ).strip()

                summary = item.get(
                    "summary",
                    ""
                ).strip()

                source = item.get(
                    "source",
                    "Unknown"
                )

                url = item.get(
                    "url",
                    ""
                )

                if not headline:
                    continue

                raw_articles.append({

                    "headline": headline,

                    "summary": summary,

                    "source": source,

                    "url": url
                })

            # -----------------------------
            # Filter relevant articles
            # -----------------------------
            articles = self.filter_articles(

                raw_articles,

                symbol=symbol,

                asset_name=asset_name
            )

            # -----------------------------
            # Limit final results
            # -----------------------------
            articles = articles[
                :max_articles
            ]

            print(

                f"[NewsFetcher] "

                f"Fetched {len(articles)} "

                f"relevant articles"
            )

            return articles

        except Exception as e:

            print(
                f"[NewsFetcher] Error: {e}"
            )

            return []

    # -----------------------------------
    # Build context for fusion
    # -----------------------------------
    def build_news_context(
        self,
        articles
    ):

        if not articles:

            return ""

        lines = []

        for article in articles:

            headline = article["headline"]

            summary = article["summary"]

            source = article["source"]

            text = (

                f"- {headline}\n"

                f"  Source: {source}\n"

                f"  Summary: {summary}"
            )

            lines.append(text)

        return "\n\n".join(lines)