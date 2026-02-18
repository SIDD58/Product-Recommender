from langchain_openai import OpenAIEmbeddings
import hashlib
from dotenv import load_dotenv
load_dotenv()

class OpenAIEmbeddingProvider:

    def __init__(self):
        self.model = "text-embedding-3-small"
        self.embeddings = OpenAIEmbeddings(
            model=self.model
        )

    def generate_key(self, text: str, prefix: str = "emb") -> str:
        """Creates a unique hash key for Redis."""
        clean_text = text.strip().lower()
        hash_val = hashlib.md5(clean_text.encode()).hexdigest()
        return f"{prefix}:{hash_val}"

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string — LangChain method 1."""
        return self.embeddings.embed_query(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of documents — LangChain method 2."""
        return self.embeddings.embed_documents(texts)