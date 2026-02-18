from dotenv import load_dotenv
from fastapi import FastAPI ,Request
import os, requests,json
import logging

from router import product_routes

server=FastAPI(title="Product Recommender API", description="API for recommending products based on user queries and product data.", version="1.0.0")
server.include_router(router=product_routes.router, prefix="/api")