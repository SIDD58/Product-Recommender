from openai import OpenAI
import os
from dotenv import load_dotenv
load_dotenv()
import hashlib 


class OpenAIEmbeddingProvider:

    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = "text-embedding-3-small"

    def generate_key(self, text: str, prefix: str = "emb") -> str:
        """Creates a unique hash key for Redis."""
        clean_text = text.strip().lower()
        hash_val = hashlib.md5(clean_text.encode()).hexdigest()
        return f"{prefix}:{hash_val}"

    def embed(self, texts: list[str]) -> list[list[float]]:
        "Direct call to openai"
        response = self.client.embeddings.create(
            model=self.model,
            input=texts
        )
        return [item.embedding for item in response.data]