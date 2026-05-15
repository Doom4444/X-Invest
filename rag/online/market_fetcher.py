import requests
import os

from dotenv import load_dotenv

load_dotenv()


class MarketFetcher:

    def __init__(self):

        self.timeout = 5

        # -----------------------------------
        # API Keys
        # -----------------------------------
        self.finnhub_key = os.getenv(
            "FINNHUB_API_KEY"
        )

        self.twelvedata_key = os.getenv(
            "TWELVEDATA_API_KEY"
        )

    # -----------------------------------
    # Finnhub
    # -----------------------------------
    def fetch_finnhub(self, symbol):

        try:

            url = (
                f"https://finnhub.io/api/v1/quote"
                f"?symbol={symbol}"
                f"&token={self.finnhub_key}"
            )

            response = requests.get(

                url,

                timeout=self.timeout
            )

            data = response.json()

            price = data.get("c")

            if not price:

                return None

            return {

                "source": "finnhub",

                "symbol": symbol,

                "price": float(price)
            }

        except Exception as e:

            print(
                f"[Finnhub] Error: {e}"
            )

            return None

    # -----------------------------------
    # TwelveData
    # -----------------------------------
    def fetch_twelvedata(self, symbol):

        try:

            url = (
                "https://api.twelvedata.com/price"
                f"?symbol={symbol}"
                f"&apikey={self.twelvedata_key}"
            )

            response = requests.get(

                url,

                timeout=self.timeout
            )

            data = response.json()

            price = data.get("price")

            if not price:

                return None

            return {

                "source": "twelvedata",

                "symbol": symbol,

                "price": float(price)
            }

        except Exception as e:

            print(
                f"[TwelveData] Error: {e}"
            )

            return None

    # -----------------------------------
    # Main Fetch Pipeline
    # -----------------------------------
    def fetch_prices(self, symbol):

        results = []

        print(
            f"[MarketFetcher] Fetching {symbol}"
        )

        # -----------------------------------
        # Finnhub
        # -----------------------------------
        finnhub_result = (
            self.fetch_finnhub(symbol)
        )

        if finnhub_result:

            print(
                "[MarketFetcher] "
                "Finnhub success"
            )

            results.append(
                finnhub_result
            )

        # -----------------------------------
        # TwelveData
        # -----------------------------------
        twelvedata_result = (
            self.fetch_twelvedata(symbol)
        )

        if twelvedata_result:

            print(
                "[MarketFetcher] "
                "TwelveData success"
            )

            results.append(
                twelvedata_result
            )

        # -----------------------------------
        # No results
        # -----------------------------------
        if not results:

            print(
                "[MarketFetcher] "
                "No providers returned data"
            )

            return []

        # -----------------------------------
        # Remove duplicate prices
        # -----------------------------------
        unique_results = []

        seen = set()

        for r in results:

            key = (

                r["source"],

                r["price"]
            )

            if key not in seen:

                seen.add(key)

                unique_results.append(r)

        return unique_results