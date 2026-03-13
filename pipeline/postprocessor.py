# pipeline/postprocessor.py
#
# PURPOSE: Final cleanup of ALLaM response before sending to user.
#
# HOW IT FITS:
#   Last step in api/chat.py before JSONResponse is returned
#   Ensures disclaimer is ALWAYS present even if model forgot it
#   The system prompt asks the model to add it, but we enforce it here too

DISCLAIMER = (
    "\n\n---\n"
    "Disclaimer: This information is for educational purposes only "
    "and is not professional financial or investment advice. "
    "Please consult a licensed financial advisor before making investment decisions."
)

def process(response: str) -> str:
    response = response.strip()
    keywords = ["disclaimer", "not professional", "educational purposes",
                "financial advisor", "not financial advice",
                "not investment advice"]
    if not any(kw in response.lower() for kw in keywords):
        response += DISCLAIMER
    return response
