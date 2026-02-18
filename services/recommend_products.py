from typing import List
from schemas.product_schema import Product, RecommendedProduct

from services.explanation import generate_explanation
from providers.openai_embeddings import OpenAIEmbeddingProvider
from services.matching.deprecate_embedding_matcher import EmbeddingMatcher
# from providers.embedding_provider import OpenAIEmbeddingProvider
# from services.matching.embedding_matcher import EmbeddingMatcher

# Initialize matcher with embedding provider
embedding_provider = OpenAIEmbeddingProvider()

matcher = EmbeddingMatcher(embedding_provider)

def recommend_products(query: str, products: List[Product]) -> List[RecommendedProduct]:
    # mock ranking (temporary)
    # Also handle case when products is empty
    top_k=1
    if not products: 
        return []
    # Also handle case when procuts are less than equal to 3
    if len(products) <= top_k:
        ranked = matcher.rank(query, products) 
    else:
        ranked = matcher.rank(query, products)[:top_k] # get top 3
    
    results: List[RecommendedProduct] = []
    for product, score in ranked:
        explanation = generate_explanation(query, product, score)
        print(f"Product: {product.title}, Score: {score}, Explanation: {explanation}")
    
        results.append(
            RecommendedProduct(
                id=product.id,
                title=product.title,
                reason=explanation
            )
        )

    return results