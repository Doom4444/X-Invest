# 
-------------------------------------------------------
from rag.ai.router import Router


router = Router()


# ===================================
# Forecast Test Cases
# ===================================
FORECAST_TESTS = [

    # -----------------------------------
    # Basic forecast
    # -----------------------------------
    "Predict Tesla",

    "Predict Apple",

    "Forecast Bitcoin",

    "Future outlook for oil prices",

    # -----------------------------------
    # Date-specific forecast
    # -----------------------------------
    "Predict Tesla on May 20 2026",

    "Forecast Apple stock price next week",

    "Bitcoin prediction tomorrow",

    "Forecast oil prices next month",

    # -----------------------------------
    # Mixed questions
    # -----------------------------------
    "Why is Tesla falling and what is the future outlook?",

    "Latest Apple news and future prediction",

    "What is happening with Bitcoin and where is it headed?",

    # -----------------------------------
    # Edge cases
    # -----------------------------------
    "Predict ABCXYZ",

    "Future price of random asset",

    "Forecast unknown company stock",

    # -----------------------------------
    # Stress tests
    # -----------------------------------
    "Can you predict Tesla stock movement for the coming days?",

    "What is the expected return for Apple?",

    "Show forecast confidence for Bitcoin",

    "What trend is expected for Tesla stock?"
]


# ===================================
# Run Forecast Tests
# ===================================
for question in FORECAST_TESTS:

    print("\n" + "=" * 80)

    print(f"QUESTION: {question}")

    print("=" * 80)

    try:

        result = router.route(question)

        print("\nANSWER:\n")

        print(
            result.get(
                "answer",
                "No answer"
            )
        )

        print("\nCONFIDENCE:")

        print(
            result.get(
                "confidence",
                "N/A"
            )
        )

        print("\nINTENT:")

        print(
            result.get(
                "intent",
                "N/A"
            )
        )

        context = result.get(
            "context",
            ""
        )

        print("\nCONTEXT LENGTH:")

        print(len(context))

        print("\nCONTEXT PREVIEW:\n")

        preview = context[:1000]

        print(preview)

        if len(context) > 1000:

            print(
                "\n...[TRUNCATED]..."
            )

    except KeyboardInterrupt:

        print(
            "\n[STOPPED BY USER]"
        )

        break

    except Exception as e:

        print(
            f"\n[ERROR] {e}"
        )