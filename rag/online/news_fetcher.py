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
    # Fetch recent news
    # -----------------------------------
    def fetch_news(
        self,
        symbol,
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
            # Clean articles
            # -----------------------------
            articles = []

            for item in data[:max_articles]:

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

                articles.append({

                    "headline": headline,

                    "summary": summary,

                    "source": source,

                    "url": url
                })

            print(
                f"[NewsFetcher] "
                f"Fetched {len(articles)} articles"
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
    