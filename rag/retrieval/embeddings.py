import ollama


class EmbeddingModel:

    def __init__(self, model_name="nomic-embed-text"):
        self.model_name = model_name

    def embed(self, texts):

        embeddings = []

        for text in texts:

            response = ollama.embeddings(
                model=self.model_name,
                prompt=text
            )

            embeddings.append(response["embedding"])

        return embeddings