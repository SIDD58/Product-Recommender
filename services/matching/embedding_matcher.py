import numpy as np
from typing import List, Tuple
from schemas.product_schema import Product
from services.matching.base import BaseMatcher
from providers.embedding_provider_langchain import OpenAIEmbeddingProvider
from services.text_builder import product_to_text
from dotenv import load_dotenv
import redis,json

load_dotenv()


class EmbeddingMatcher(BaseMatcher):

    def __init__(self, embedding_provider: OpenAIEmbeddingProvider):
        self.embedding_provider = embedding_provider

    def rank(self, query: str, products: List[Product]) -> List[Tuple[Product, float]]:
        """
        Returns products sorted by relevance score using cosine similarity
        between the query embedding and each product embedding.
        """
        if not products:
            return []

        # Build product text representations
        product_texts = [product_to_text(p) for p in products]

        # Single batch call (query + all products together) This minimize API calls 

        query_embedding = np.array(self.embedding_provider.embed_query(query))
        product_embeddings = np.array(self.embedding_provider.embed_documents(product_texts))

        # Cosine similarity between query and each product
        scores = self._cosine_similarity(query_embedding, product_embeddings)

        # Pair each product with its score and sort descending
        ranked = sorted(
            zip(products, scores.tolist()),
            key=lambda x: x[1],
            reverse=True
        )

        return ranked

    # def _product_to_text(self, product: Product) -> str:
    #     """
    #     Converts a Product into a single string for embedding.
    #     Adjust fields based on your Product schema.
    #     """
    #     parts = [
    #         product.name,
    #         product.description or "",
    #         product.category or "",
    #     ]
    #     return " | ".join(filter(None, parts))

    def _cosine_similarity(self, query_vec: np.ndarray, product_vecs: np.ndarray) -> np.ndarray:
        """
        Computes cosine similarity between a single query vector
        and a matrix of product vectors.
        """
        query_norm = query_vec / np.linalg.norm(query_vec)
        product_norms = product_vecs / np.linalg.norm(product_vecs, axis=1, keepdims=True)
        return product_norms @ query_norm