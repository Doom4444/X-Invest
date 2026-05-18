# rag/embeddings.py

import ollama

from typing import List, Union


class EmbeddingModel:

    def __init__(

        self,

        model_name: str = "bge-m3"
    ):

        self.model_name = model_name

        # -----------------------------------
        # Simple in-memory cache
        # -----------------------------------
        self._cache = {}

    # -----------------------------------
    # Embed single text
    # -----------------------------------
    def _embed_one(self, text: str):

        if text in self._cache:

            return self._cache[text]

        response = ollama.embeddings(

            model=self.model_name,

            prompt=text
        )

        emb = response["embedding"]

        self._cache[text] = emb

        return emb

    # -----------------------------------
    # Embed single or multiple texts
    # -----------------------------------
    def embed(
        self,
        texts: Union[str, List[str]]
    ):

        # -----------------------------------
        # Single text
        # -----------------------------------
        if isinstance(texts, str):

            return [
                self._embed_one(texts)
            ]

        # -----------------------------------
        # Multiple texts
        # -----------------------------------
        return [

            self._embed_one(t)

            for t in texts
        ]