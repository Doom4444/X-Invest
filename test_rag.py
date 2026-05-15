from rag.ai.router import Router


router = Router()


# ===================================
# Structured Test Cases
# ===================================
TEST_CASES = {

    # -----------------------------------
    # General Finance + RAG
    # -----------------------------------
    "general_finance": [

        "What is diversification?",

        "Why is diversification important for investors?",

        "What is inflation?",

        "How is inflation measured using CPI?",

        "What is the Consumer Price Index?",

        "What is core inflation?",

        "What are corporate bonds?",

        "What is default risk in bonds?",

        "Difference between stocks and corporate bonds",

        "What are investment-grade bonds?",

        "Why do long-term bonds usually offer higher interest rates?",

        "Why is saving and investing important?",

        "How can investing help achieve financial security?",

        "What questions should investors ask financial professionals?"
    ],

    # -----------------------------------
    # Federal Reserve / Monetary Policy
    # -----------------------------------
    "fed_and_policy": [

        "What are the main functions of the Federal Reserve?",

        "What is monetary policy?",

        "How does the Federal Reserve conduct monetary policy?",

        "What are the Federal Reserve's key responsibilities?",

        "What does the Fed do to maintain financial stability?",

        "What does the Monetary Policy Report say about inflation?",

        "What does the Federal Reserve say about labor markets?",

        "What are the recent economic developments discussed in the report?"
    ],

    # -----------------------------------
    # OECD / Macroeconomics
    # -----------------------------------
    "macroeconomics": [

        "What does the OECD report say about global GDP growth in 2025?",

        "What are the risks facing the global economy according to OECD?",

        "How does the OECD describe trade policy uncertainty?",

        "What does OECD say about inflation trends?",

        "Why does OECD believe global growth is fragile?"
    ],

    # -----------------------------------
    # WIPO Financial Report
    # -----------------------------------
    "financial_reports": [

        "What happened to WIPO financial performance in 2020?",

        "How did COVID-19 affect WIPO financial operations?",

        "What financial risks were discussed in the WIPO report?"
    ],

    # -----------------------------------
    # Market Data
    # -----------------------------------
    "market_data": [

        "What is Tesla stock price today?",

        "Bitcoin price now",

        "Gold price today",

        "Current Apple stock price",

        "Latest oil price"
    ],

    # -----------------------------------
    # Forecast
    # -----------------------------------
    "forecast": [

        "Predict Tesla future stock price",

        "Forecast Bitcoin next year",

        "Future outlook for oil prices"
    ],

    # -----------------------------------
    # News Analysis
    # -----------------------------------
    "news_analysis": [

        "Why is Tesla stock falling today?",

        "Latest Bitcoin news",

        "What is affecting oil prices recently?",

        "Recent market news about Apple"
    ],

    # -----------------------------------
    # Edge Cases
    # -----------------------------------
    "edge_cases": [

        "What is ABCXYZ coin price?",

        "Predict unknown asset future",

        "Latest news about random company",

        "Tell me about some unknown economy",

        "What is the price of random asset XYZ?"
    ]
}


# ===================================
# Run All Tests
# ===================================
for category, questions in TEST_CASES.items():

    print("\n" + "#" * 80)

    print(
        f"TEST CATEGORY: "
        f"{category.upper()}"
    )

    print("#" * 80)

    for q in questions:

        print("\n" + "=" * 70)

        print(f"QUESTION: {q}")

        print("=" * 70)

        try:

            result = router.route(q)

            print("\nANSWER:\n")

            print(
                result.get("answer")
            )

            print("\nCONFIDENCE:")

            print(
                result.get("confidence")
            )

            print("\nINTENT:")

            print(
                result.get("intent")
            )

            context = result.get(
                "context",
                ""
            )

            print("\nCONTEXT LENGTH:")

            print(len(context))

            print("\nCONTEXT PREVIEW:\n")

            preview = context[:500]

            print(preview)

            if len(context) > 500:

                print(
                    "\n...[TRUNCATED]..."
                )

        except KeyboardInterrupt:

            print(
                "\n[STOPPED BY USER]"
            )

            exit()

        except Exception as e:

            print(f"\n[ERROR] {e}")