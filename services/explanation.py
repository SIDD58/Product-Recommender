from schemas.product_schema import Product
from typing import List


def _normalize(text: str):
    return set(text.lower().split())

# we create expalantion based on keyword matches (title and query) and semantic similarity score buckets


def generate_explanation(query: str, product: Product, score: float) -> str:
    query_terms = _normalize(query)
    title_terms = _normalize(product.title)
    tag_terms = set(t.lower() for t in product.tags)

    matched_terms = query_terms & title_terms
    matched_tags = query_terms & tag_terms

    parts: List[str]= []

    if matched_terms:
        parts.append(f"matches keywords {', '.join(sorted(matched_terms))}")

    if matched_tags:
        parts.append(f"matches category {', '.join(sorted(matched_tags))}")

    # semantic interpretation bucket (deterministic)
    if score > 0.85:
        parts.append("very high semantic similarity")
    elif score > 0.70:
        parts.append("high semantic similarity")
    elif score > 0.55:
        parts.append("moderate semantic similarity")
    else:
        parts.append("loosely related")

    return "; ".join(parts).capitalize() + "."