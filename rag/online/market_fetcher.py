import requests
import concurrent.futures
import time
import os
from dotenv import load_dotenv

load_dotenv()

finnhub_key = os.getenv("FINNHUB_API_KEY")
twelvedata_key = os.getenv("TWELVEDATA_API_KEY")


class MarketFetcher:

    def __init__(self):

        self.timeout = 5
        self.max_retries = 3

        # read API keys from .env
        self.finnhub_key = finnhub_key
        self.twelvedata_key = twelvedata_key