class MarketVerifier:

    def __init__(self):

        # Maximum acceptable deviation
        self.max_deviation = 0.05

    # -----------------------------------
    # Verify market results
    # -----------------------------------
    def verify(self, results):

        # No results
        if not results:

            return None, 0.0

        # Extract prices
        prices = [
            r["price"]
            for r in results
            if r.get("price") is not None
        ]

        # No valid prices
        if not prices:

            return None, 0.0

        # -----------------------------------
        # Single source
        # -----------------------------------
        if len(prices) == 1:

            price = round(prices[0], 2)

            confidence = 65.0

            return price, confidence

        # -----------------------------------
        # Multi-source average
        # -----------------------------------
        avg_price = sum(prices) / len(prices)

        # Largest deviation
        max_diff = max(
            abs(p - avg_price)
            for p in prices
        )

        # Relative deviation
        deviation = max_diff / avg_price

        # -----------------------------------
        # Confidence calculation
        # -----------------------------------
        if deviation < 0.01:

            confidence = 95.0

        elif deviation < 0.03:

            confidence = 85.0

        elif deviation < self.max_deviation:

            confidence = 70.0

        else:

            confidence = 50.0

        # Final rounded price
        final_price = round(avg_price, 2)

        print(
            f"[Verifier] Avg Price: {final_price}"
        )

        print(
            f"[Verifier] Deviation: "
            f"{round(deviation, 4)}"
        )

        print(
            f"[Verifier] Confidence: "
            f"{confidence}%"
        )

        return final_price, confidence