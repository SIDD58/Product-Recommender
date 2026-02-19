from fastapi import FastAPI

#from router import product_routes
from router import async_product_routes

server=FastAPI(title="Product Recommender API", description="API for recommending products based on user queries and product data.", version="1.0.0")
#server.include_router(router=product_routes.router, prefix="/api")
server.include_router(router=async_product_routes.router, prefix="/api")