from rag.ai.router import Router


router = Router()

questions = [

    # -----------------------------
    # Investing Basics
    # -----------------------------
    "What is diversification?",

    "Why is diversification important for investors?",

    "What are the keys to financial success?",

    # -----------------------------
    # Inflation
    # -----------------------------
    "What is inflation?",

    "How is inflation measured?",

    "What is CPI?",

    # -----------------------------
    # Bonds
    # -----------------------------
    "What are corporate bonds?",

    "What is default risk in bonds?",

    "Difference between stocks and bonds",

    # -----------------------------
    # Federal Reserve
    # -----------------------------
    "What does the Federal Reserve do?",

    "What is monetary policy?",

    "How does the Fed control inflation?",

    # -----------------------------
    # Market Data
    # -----------------------------
    "What is Tesla stock price today?",

    "Bitcoin price now",

    "Gold price today",

    # -----------------------------
    # Forecast
    # -----------------------------
    "Predict Tesla future stock price",

    # -----------------------------
    # Macroeconomics
    # -----------------------------
    "What does the OECD say about global growth?",

    "What are the risks facing the global economy?",

    # -----------------------------
    # Financial Reports
    # -----------------------------
    "What happened to WIPO financial performance in 2020?"
]

for q in questions:

    print("\n" + "=" * 70)

    print(f"QUESTION: {q}")

    print("=" * 70)

    try:

        result = router.route(q)

        print("\nANSWER:\n")

        print(result.get("answer"))

        print("\nCONFIDENCE:")

        print(result.get("confidence"))

        print("\nINTENT:")

        print(result.get("intent"))

    except Exception as e:

        print(f"\n[ERROR] {e}")