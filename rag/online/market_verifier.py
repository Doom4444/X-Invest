class MarketVerifier:

    def __init__(self):

        # الحد الأقصى للاختلاف المقبول
        self.max_deviation = 0.05

    def verify(self, results):

        if not results:
            return None, "0%"

        prices = [r["price"] for r in results]

        # لو مصدر واحد فقط
        if len(prices) == 1:

            price = round(prices[0], 2)

            confidence = "65%"

            return price, confidence

        # حساب المتوسط
        avg_price = sum(prices) / len(prices)

        # أكبر فرق
        max_diff = max(abs(p - avg_price) for p in prices)

        # نسبة الاختلاف
        deviation = max_diff / avg_price

        # حساب الثقة
        if deviation < 0.01:

            confidence = 95

        elif deviation < 0.03:

            confidence = 85

        elif deviation < self.max_deviation:

            confidence = 70

        else:

            confidence = 50

        price = round(avg_price, 2)

        return price, f"{confidence}%"