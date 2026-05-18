# -------------------------------------------------------
# Full System Evaluation Suite
# -------------------------------------------------------

from rag.ai.router import Router


router = Router()


# ===================================
# 1. Intent Routing Tests
# ===================================
INTENT_TESTS = [

    # --------------------------------
    # Forecast intent
    # --------------------------------
    "Predict Tesla",

    "Forecast Bitcoin tomorrow",

    "Tesla future outlook",

    # --------------------------------
    # Market intent
    # --------------------------------
    "What is Tesla price now?",

    "Current Bitcoin price",

    "How much is Apple stock?",

    # --------------------------------
    # News / analysis intent
    # --------------------------------
    "Why is Tesla falling?",

    "Latest Apple news",

    "What is happening with Bitcoin?",

    # --------------------------------
    # General finance
    # --------------------------------
    "What is diversification?",

    "Explain market volatility",

    "What is risk management?"
]


# ===================================
# 2. Asset Extraction Tests
# ===================================
ASSET_TESTS = [

    # --------------------------------
    # Rule-based extraction
    # --------------------------------
    "Predict Tesla",

    "Forecast Apple",

    "Bitcoin prediction",

    # --------------------------------
    # Regex extraction
    # --------------------------------
    "Predict TSLA",

    "What about AAPL next week?",

    "Forecast BTC-USD",

    "Should I buy NVDA?",

    # --------------------------------
    # Unknown assets
    # --------------------------------
    "Predict ABCXYZ",

    "Future of RANDOMCOIN",

    "Forecast unknown company"
]


# ===================================
# 3. Forecast Tests
# ===================================
FORECAST_TESTS = [

    # --------------------------------
    # Basic forecast
    # --------------------------------
    "Predict Tesla",

    "Predict Apple",

    "Forecast Bitcoin",

    # --------------------------------
    # Date-specific forecast
    # --------------------------------
    "Predict Tesla on May 20 2026",

    "Bitcoin prediction tomorrow",

    "Forecast Apple next week",

    # --------------------------------
    # Forecast metrics
    # --------------------------------
    "Show forecast confidence for Tesla",

    "Expected return for Apple",

    "Tesla forecast trend"
]


# ===================================
# 4. RAG Retrieval Tests
# ===================================
RAG_TESTS = [

    # --------------------------------
    # Educational finance
    # --------------------------------
    "What is inflation?",

    "Explain ETFs",

    "What is portfolio diversification?",

    # --------------------------------
    # Complex retrieval
    # --------------------------------
    "How does compound interest affect investments?",

    "What are the risks of cryptocurrency investing?",

    "Explain support and resistance levels"
]


# ===================================
# 5. Multi-Source Fusion Tests
# ===================================
FUSION_TESTS = [

    # --------------------------------
    # Forecast + News
    # --------------------------------
    "Why is Tesla falling and what is the future outlook?",

    "Latest Apple news and future prediction",

    # --------------------------------
    # Market + Forecast + News
    # --------------------------------
    "What is happening with Bitcoin and where is it headed?",

    "Analyze Tesla stock and predict the next trend",

    # --------------------------------
    # Full hybrid reasoning
    # --------------------------------
    "Should investors worry about Tesla's recent decline and future forecast?"
]


# ===================================
# 6. Hallucination Resistance Tests
# ===================================
HALLUCINATION_TESTS = [

    # --------------------------------
    # Fake assets
    # --------------------------------
    "Predict SUPERFAKECOIN",

    "Forecast XYZABC123",

    # --------------------------------
    # Impossible questions
    # --------------------------------
    "What will Tesla stock be exactly in 10 years?",

    "Can you guarantee Bitcoin profits?",

    # --------------------------------
    # Missing data
    # --------------------------------
    "Forecast unknown private company stock"
]


# ===================================
# 7. Stress Tests
# ===================================
STRESS_TESTS = [

    # --------------------------------
    # Long queries
    # --------------------------------
    (
        "Can you analyze Tesla stock "
        "considering market trends, "
        "recent news, future forecasts, "
        "and investment risks?"
    ),

    # --------------------------------
    # Multi-asset queries
    # --------------------------------
    (
        "Compare Tesla, Apple, "
        "and Bitcoin future outlooks"
    ),

    # --------------------------------
    # Ambiguous wording
    # --------------------------------
    "What do you think about Tesla?",

    "Is Bitcoin safe?",

    "Should I invest now?"
]


# ===================================
# All Test Groups
# ===================================
ALL_TEST_GROUPS = {

    "Intent Tests":
        INTENT_TESTS,

    "Asset Extraction Tests":
        ASSET_TESTS,

    "Forecast Tests":
        FORECAST_TESTS,

    "RAG Tests":
        RAG_TESTS,

    "Fusion Tests":
        FUSION_TESTS,

    "Hallucination Tests":
        HALLUCINATION_TESTS,

    "Stress Tests":
        STRESS_TESTS
}


# ===================================
# Save Results
# ===================================
with open(

    "test_results.txt",

    "w",

    encoding="utf-8"

) as f:

    for group_name, tests in ALL_TEST_GROUPS.items():

        separator = "=" * 80

        f.write("\n")
        f.write(separator + "\n")

        f.write(group_name.upper() + "\n")

        f.write(separator + "\n")

        for question in tests:

            print(
                f"\nRunning: {question}"
            )

            f.write("\n")
            f.write("-" * 80 + "\n")

            f.write(
                f"QUESTION: {question}\n"
            )

            f.write("-" * 80 + "\n")

            try:

                result = router.route(
                    question
                )

                f.write("\nANSWER:\n\n")

                f.write(
                    result["answer"] + "\n"
                )

                f.write(
                    "\nCONFIDENCE:\n"
                )

                f.write(
                    str(
                        result[
                            "confidence"
                        ]
                    ) + "\n"
                )

                f.write("\nINTENT:\n")

                f.write(
                    result["intent"] + "\n"
                )

                f.write(
                    "\nCONTEXT LENGTH:\n"
                )

                f.write(
                    str(
                        len(
                            result[
                                "context"
                            ]
                        )
                    ) + "\n"
                )

                f.write(
                    "\nCONTEXT PREVIEW:\n\n"
                )

                f.write(
                    result[
                        "context"
                    ][:2000]
                )

                f.write("\n\n")

            except Exception as e:

                f.write(
                    f"\nERROR: {e}\n"
                )

                f.write("\n\n")

print(
    "\n✅ Results saved to "
    "test_results.txt"
)