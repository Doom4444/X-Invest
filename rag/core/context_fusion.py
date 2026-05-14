class ContextFusion:

    def __init__(self):

        # max chars per section
        self.max_section_length = 2500

    # -----------------------------------
    # Remove duplicate texts
    # -----------------------------------
    def deduplicate(self, items):

        unique_items = []

        seen = set()

        for item in items:

            clean_item = item.strip()

            if not clean_item:
                continue

            # normalized comparison
            normalized = clean_item.lower()

            if normalized not in seen:

                seen.add(normalized)

                unique_items.append(clean_item)

        return unique_items

    # -----------------------------------
    # Trim very large context
    # -----------------------------------
    def compress(self, text):

        if len(text) <= self.max_section_length:

            return text

        return text[:self.max_section_length] + "..."

    # -----------------------------------
    # Build structured section
    # -----------------------------------
    def build_section(self, title, content):

        if not content.strip():

            return ""

        return (
            f"[{title}]\n"
            f"{content}"
        )

    # -----------------------------------
    # Main Fusion Pipeline
    # -----------------------------------
    def fuse(

        self,

        rag_context="",

        market_context="",

        news_context="",

        prediction_context=""
    ):

        sections = []

        # -----------------------------------
        # RAG Section
        # -----------------------------------
        if rag_context:

            rag_chunks = rag_context.split("\n\n")

            rag_chunks = self.deduplicate(
                rag_chunks
            )

            rag_text = "\n\n".join(
                rag_chunks
            )

            rag_text = self.compress(
                rag_text
            )

            sections.append(

                self.build_section(
                    "RAG KNOWLEDGE",
                    rag_text
                )
            )

        # -----------------------------------
        # Market Section
        # -----------------------------------
        if market_context:

            sections.append(

                self.build_section(
                    "MARKET DATA",
                    market_context
                )
            )

        # -----------------------------------
        # News Section
        # -----------------------------------
        if news_context:

            sections.append(

                self.build_section(
                    "NEWS ANALYSIS",
                    news_context
                )
            )

        # -----------------------------------
        # Prediction Section
        # -----------------------------------
        if prediction_context:

            sections.append(

                self.build_section(
                    "PREDICTION",
                    prediction_context
                )
            )

        # -----------------------------------
        # Final Fusion
        # -----------------------------------
        fused_context = "\n\n".join(

            section

            for section in sections

            if section.strip()
        )

        return fused_context