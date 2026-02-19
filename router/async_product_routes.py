from fastapi import APIRouter,HTTPException
# from app.schemas.product import RecommendationRequest, RecommendationResponse
# from app.services.recommender import recommend_products
from services.async_recommend_products import recommend_products
from schemas.product_schema import RecommendationRequest, RecommendationResponse

router = APIRouter()


@router.post("/recommend", response_model=RecommendationResponse)
async def recommend(data: RecommendationRequest):
    try:
        results = await recommend_products(data.query, data.products)
        return {"recommendations": results} 
    except Exception as e: 
        raise HTTPException(status_code=500, detail=str(e))

