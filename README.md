
# How to run the code 
docker compose up -d

uv run uvicorn main:server --reload



# testing

Post Request URL : http://127.0.0.1:8000/api/recommend

INPUT body (JSON)

{
   "query": "red party dress",
   "products": [
     { "id": 1, "title": "Red Sequin Gown", "tags": ["party", "evening"]
},
     { "id": 2, "title": "Black Cocktail Dress", "tags": ["cocktail"] },
     { "id": 3, "title": "Red Maxi Dress", "tags": ["casual"] }
   ]
}