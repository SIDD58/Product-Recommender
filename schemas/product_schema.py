from pydantic import BaseModel
from typing import List


# Product 
class Product(BaseModel):
    id: int
    title: str
    tags: List[str]

# request with product list and query
class RecommendationRequest(BaseModel):
    query: str
    products: List[Product]


# signals for matching query and product
# class MatchSignals(BaseModel):  
#     semantic_score: float
#     matched_terms: List[str]
#     matched_tags: List[str]

# recommended product with reason for recommendation
class RecommendedProduct(BaseModel):
    id: int
    title: str
    reason: str
    # score: float #score
    # explanation: str #expalantion
    # signals: MatchSignals #signal 


class RecommendationResponse(BaseModel):
    recommendations: List[RecommendedProduct]
