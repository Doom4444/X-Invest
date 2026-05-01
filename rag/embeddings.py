# rag/embeddings.py
import ollama
from typing import List, Union


class EmbeddingModel:
    def __init__(self, model_name: str = "nomic-embed-text"):
        self.model_name = model_name
        self._cache = {}  # simple in-memory cache

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

    def embed(self, texts: Union[str, List[str]]):
        # accept single string or list
        if isinstance(texts, str):
            return [self._embed_one(texts)]

        return [self._embed_one(t) for t in texts]