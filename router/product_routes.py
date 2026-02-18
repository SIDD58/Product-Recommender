from fastapi import APIRouter
# from app.schemas.product import RecommendationRequest, RecommendationResponse
# from app.services.recommender import recommend_products
from services.recommend_products import recommend_products
from schemas.product_schema import RecommendationRequest, RecommendationResponse

router = APIRouter()


@router.post("/recommend", response_model=RecommendationResponse)
def recommend(data: RecommendationRequest):
    results = recommend_products(data.query, data.products)
    return {"recommendations": results}