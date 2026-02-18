from typing import List
from schemas.product_schema import Product, RecommendedProduct
from services.explanation import generate_explanation
from providers.openai_embedding_provider import OpenAIEmbeddingProvider
from services.matching.openai_embedding_matcher import EmbeddingMatcher
# from providers.embedding_provider import OpenAIEmbeddingProvider
# from services.matching.embedding_matcher import EmbeddingMatcher

# Initialize matcher with embedding provider
embedding_provider = OpenAIEmbeddingProvider()

matcher = EmbeddingMatcher(embedding_provider)

def recommend_products(query: str, products: List[Product]) -> List[RecommendedProduct]:
    # Also handle case when products is empty
    top_k=3
    is_fallback = False
    if not products: 
        return []
    # Also handle case when procuts are less than equal to k
    all_ranked=matcher.rank(query,products)
    try:
        # Attempt AI Ranking
        # raise Exception("An intentional, generic exception so fall back can be executed")
        ranked = all_ranked=matcher.rank(query,products)
    except Exception as e:
        # In case of failure Jaccard logic implementation
        print(f"Semantic match failed: {e}. Falling back to Jaccard Similarity.")
        ranked = matcher.keyword_fallback(query, products)
        is_fallback = True



    if len(products) <= top_k:
        ranked = all_ranked
    else:
        ranked = all_ranked[:top_k] # get top k
    
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