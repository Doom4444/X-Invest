class ContextFusion:

    def __init__(self):

        # -----------------------------------
        # Max chars per section
        # -----------------------------------
        self.max_section_length = 2500

        # -----------------------------------
        # Max RAG chunks
        # -----------------------------------
        self.max_rag_chunks = 3

    # -----------------------------------
    # Normalize text
    # -----------------------------------
    def normalize_text(self, text):

        return (

            text.lower()

            .strip()

            .replace("\n", " ")
        )

    # -----------------------------------
    # Jaccard similarity
    # -----------------------------------
    def jaccard_similarity(

        self,

        text1,

        text2
    ):

        words1 = set(
            text1.split()
        )

        words2 = set(
            text2.split()
        )

        intersection = len(
            words1 & words2
        )

        union = len(
            words1 | words2
        )

        if union == 0:

            return 0

        return intersection / union

    # -----------------------------------
    # Remove duplicate texts
    # -----------------------------------
    def deduplicate(

        self,

        items,

        similarity_threshold=0.85
    ):

        unique_items = []

        normalized_seen = []

        for item in items:

            clean_item = item.strip()

            if not clean_item:

                continue

            normalized = self.normalize_text(
                clean_item
            )

            duplicate_found = False

            for existing in normalized_seen:

                similarity = (

                    self.jaccard_similarity(

                        normalized,

                        existing
                    )
                )

                if similarity >= similarity_threshold:

                    duplicate_found = True

                    break

            if duplicate_found:

                continue

            normalized_seen.append(
                normalized
            )

            unique_items.append(
                clean_item
            )

        return unique_items

    # -----------------------------------
    # Trim very large context
    # -----------------------------------
    def compress(self, text):

        if len(text) <= self.max_section_length:

            return text

        trimmed = text[
            :self.max_section_length
        ]

        # --------------------------------
        # Sentence-safe compression
        # --------------------------------
        last_period = trimmed.rfind(".")

        # --------------------------------
        # Fallback to newline
        # --------------------------------
        if last_period <= 100:

            last_period = trimmed.rfind(
                "\n"
            )

        if last_period > 100:

            trimmed = trimmed[
                :last_period + 1
            ]

        return trimmed + "..."

    # -----------------------------------
    # Build structured section
    # -----------------------------------
    def build_section(

        self,

        title,

        content
    ):

        if not content.strip():

            return ""

        return (
            f"[{title}]\n"
            f"{content}"
        )

    # -----------------------------------
    # Build sections dynamically
    # -----------------------------------
    def build_sections(

        self,

        rag_context="",

        market_context="",

        news_context="",

        prediction_context=""
    ):

        sections = {

            "rag": "",

            "market": "",

            "news": "",

            "forecast": ""
        }

        # -----------------------------------
        # Market Section
        # -----------------------------------
        if market_context.strip():

            sections["market"] = (

                self.build_section(
                    "MARKET DATA",
                    market_context
                )
            )

        # -----------------------------------
        # Forecast Section
        # -----------------------------------
        if prediction_context.strip():

            sections["forecast"] = (

                self.build_section(
                    "FORECAST",
                    prediction_context
                )
            )

        # -----------------------------------
        # News Section
        # -----------------------------------
        if news_context.strip():

            news_text = self.compress(
                news_context
            )

            sections["news"] = (

                self.build_section(
                    "NEWS ANALYSIS",
                    news_text
                )
            )

        # -----------------------------------
        # RAG Section
        # -----------------------------------
        if rag_context.strip():

            rag_chunks = rag_context.split(
                "\n\n"
            )

            # --------------------------------
            # Deduplicate chunks
            # --------------------------------
            rag_chunks = self.deduplicate(
                rag_chunks
            )

            # --------------------------------
            # Limit chunk count
            # --------------------------------
            rag_chunks = rag_chunks[
                :self.max_rag_chunks
            ]

            rag_text = "\n\n".join(
                rag_chunks
            )

            rag_text = self.compress(
                rag_text
            )

            sections["rag"] = (

                self.build_section(
                    "RAG KNOWLEDGE",
                    rag_text
                )
            )

        return sections

    # -----------------------------------
    # Intent-aware priority
    # -----------------------------------
    def get_priority_order(
        self,
        intent
    ):

        priority_map = {

            # --------------------------------
            # Forecast-heavy queries
            # --------------------------------
            "forecast": [

                "forecast",

                "market",

                "news",

                "rag"
            ],

            # --------------------------------
            # Market-focused queries
            # --------------------------------
            "market_data": [

                "market",

                "forecast",

                "news",

                "rag"
            ],

            # --------------------------------
            # News / analysis queries
            # --------------------------------
            "analysis": [

                "news",

                "market",

                "forecast",

                "rag"
            ],

            # --------------------------------
            # General finance questions
            # --------------------------------
            "general_finance": [

                "rag",

                "news",

                "market",

                "forecast"
            ]
        }

        # -----------------------------------
        # Default fallback
        # -----------------------------------
        return priority_map.get(

            intent,

            [

                "market",

                "forecast",

                "news",

                "rag"
            ]
        )

    # -----------------------------------
    # Main Fusion Pipeline
    # -----------------------------------
    def fuse(

        self,

        intent="general_finance",

        rag_context="",

        market_context="",

        news_context="",

        prediction_context=""
    ):

        # -----------------------------------
        # Build all sections
        # -----------------------------------
        sections = self.build_sections(

            rag_context=rag_context,

            market_context=market_context,

            news_context=news_context,

            prediction_context=prediction_context
        )

        # -----------------------------------
        # Dynamic priority ordering
        # -----------------------------------
        priority_order = (
            self.get_priority_order(
                intent
            )
        )

        ordered_sections = []

        for section_name in priority_order:

            section = sections.get(
                section_name,
                ""
            )

            if section.strip():

                ordered_sections.append(
                    section
                )

        # -----------------------------------
        # Final Fusion
        # -----------------------------------
        fused_context = "\n\n".join(

            ordered_sections
        )

        return fused_context