from typing import List, Tuple
from schemas.product_schema import Product
from services.matching.base import BaseMatcher
from services.text_builder import product_to_text
from providers.openai_embedding_provider import OpenAIEmbeddingProvider
from dotenv import load_dotenv
import numpy as np
import redis,os,json
load_dotenv()
redis_url=os.environ.get('REDIS_URL','redis://localhost:6379/0')


class EmbeddingMatcher(BaseMatcher):

    # intialize embedding provider and redis
    def __init__(self, embedding_provider: OpenAIEmbeddingProvider):
        self.embedding_provider = embedding_provider
        self.redis = redis.from_url(redis_url)  # your existing redis_url


    # prelaod all the products whose embeddings are not cached 
    def preload(self, products: List[Product]) -> None:
        """Call once at startup to pre-compute and cache all product embeddings in Redis."""
        uncached = [p for p in products if not self.redis.exists(self._key(p))]
        if not uncached:
            print("All product embeddings already cached in Redis")
            return

        texts = [product_to_text(p) for p in uncached]
        embeddings = self.embedding_provider.embed(texts)

        # Batch write to Redis using pipeline for speed, (Less Network round trip : Batch operation)
        pipe = self.redis.pipeline()
        for product, embedding in zip(uncached, embeddings):
            pipe.set(self._key(p), json.dumps(embedding))
        pipe.execute()

        print(f"Preloaded {len(uncached)} product embeddings into Redis")

    # Ranking Logic 
    def rank(self, query: str, products: List[Product]) -> List[Tuple[Product, float]]:
        if not products:
            return []

        # Step 1 — Batch fetch from Redis
        pipe = self.redis.pipeline()
        for p in products:
            pipe.get(self._key(p))
        cached_results = pipe.execute()

        # Step 2 — Find missing ones
        uncached = [
            p for p, result in zip(products, cached_results)
            if result is None
        ]

        # Step 3 — Embed and store missing
        if uncached:
            texts = [product_to_text(p) for p in uncached]
            embeddings = self.embedding_provider.embed(texts)

            pipe = self.redis.pipeline()
            for product, embedding in zip(uncached, embeddings):
                pipe.set(self._key(product), json.dumps(embedding))
            pipe.execute()

            # Build a lookup so we don't need a second Redis fetch
            newly_cached = {
                self._key(p): json.dumps(emb)
                for p, emb in zip(uncached, embeddings)
            }

            # Merge into cached_results — replace None with newly computed embeddings
            cached_results = [
                result if result is not None else newly_cached[self._key(p)]
                for p, result in zip(products, cached_results)
            ]

        # Step 4 — Safe parse, skip any still-None (shouldn't happen now)
        product_embeddings = np.array([json.loads(r) for r in cached_results if r is not None])
        query_embedding = np.array(self.embedding_provider.embed([query])[0])

        scores = self._cosine_similarity(query_embedding, product_embeddings)

        ranked = sorted(
            zip(products, scores.tolist()),
            key=lambda x: x[1],
            reverse=True
        )
        return ranked
    

    # key for the redis Cache 
    def _key(self, product: Product) -> str:
        text = product_to_text(product)
        return self.embedding_provider.generate_key(text, prefix="emb:product")
    
    # Similarity metrics 
    def _cosine_similarity(self, query_vec: np.ndarray, product_vecs: np.ndarray) -> np.ndarray:
        query_norm = query_vec / np.linalg.norm(query_vec)
        product_norms = product_vecs / np.linalg.norm(product_vecs, axis=1, keepdims=True)
        return product_norms @ query_norm
    
    # Keyword Fallback in case openai embedding fails 
    def keyword_fallback(self, query: str, products: List[Product]) -> List[Tuple[Product, float]]:
        """A fast, local fallback using keyword overlap (Jaccard Similarity)."""
        query_tokens = set(query.lower().split())
        results = []

        for product in products:
            # Combine title and tags for matching
            product_text = f"{product.title} {' '.join(product.tags)}".lower()
            product_tokens = set(product_text.split())
            
            # Intersection (how many words match)
            intersection = query_tokens.intersection(product_tokens)
            # Union (total unique words)
            union = query_tokens.union(product_tokens)
            
            # Jaccard Score: 0.0 to 1.0
            score = len(intersection) / len(union) if union else 0.0
            results.append((product, score))

        # Sort by match count
        return sorted(results, key=lambda x: x[1], reverse=True)
    




# deprecated methods with redis 


# def rank(self, query: str, products: List[Product]) -> List[Tuple[Product, float]]:
    #     if not products:
    #         return []

    #     # Fetch all product embeddings from Redis in one batch
    #     pipe = self.redis.pipeline()
    #     for p in products:
    #         pipe.get(self._key(p))
    #     cached_results = pipe.execute()

    #     # Find which products are missing from Redis
    #     uncached = [
    #         p for p, result in zip(products, cached_results)
    #         if result is None
    #     ]

    #     # Embed and store missing products
    #     if uncached:
    #         texts = [product_to_text(p) for p in uncached]
    #         embeddings = self.embedding_provider.embed(texts)

    #         pipe = self.redis.pipeline()
    #         for product, embedding in zip(uncached, embeddings):
    #             pipe.set(self._key(p), json.dumps(embedding))
    #         pipe.execute()

    #         # Refresh cached_results with newly stored embeddings
    #         pipe = self.redis.pipeline()
    #         for p in products:
    #             pipe.get(self._key(p))
    #         cached_results = pipe.execute()

    #     # Build product embeddings matrix from Redis results
    #     product_embeddings = np.array([json.loads(r) for r in cached_results])

    #     # Only OpenAI call per request is the query
    #     query_embedding = np.array(self.embedding_provider.embed([query])[0])

    #     scores = self._cosine_similarity(query_embedding, product_embeddings)

    #     ranked = sorted(
    #         zip(products, scores.tolist()),
    #         key=lambda x: x[1],
    #         reverse=True
    #     )
    #     return ranked



    # def _key(self, product_id) -> str:
    #     """Generates a Redis key for a product embedding."""
    #     return f"emb:product:{product_id}"

# Memory cache No redis
# class EmbeddingMatcher(BaseMatcher):

#     def __init__(self, embedding_provider: OpenAIEmbeddingProvider):
#         self.embedding_provider = embedding_provider
#         self._product_cache: dict[str, list[float]] = {}  # product_id → embedding

#     def preload(self, products: List[Product]) -> None:
#         """Call once at startup to pre-compute all product embeddings."""
#         uncached = [p for p in products if p.id not in self._product_cache]
#         if not uncached:
#             return

#         texts = [self._product_to_text(p) for p in uncached]
#         embeddings = self.embedding_provider.embed(texts)  # single batch call

#         for product, embedding in zip(uncached, embeddings):
#             self._product_cache[product.id] = embedding

#         print(f"Preloaded {len(uncached)} product embeddings")

#     def rank(self, query: str, products: List[Product]) -> List[Tuple[Product, float]]:
#         if not products:
#             return []

#         # Embed any products not yet in cache
#         uncached = [p for p in products if str(p.id) not in self._product_cache]
#         if uncached:
#             texts = [product_to_text(p) for p in uncached]
#             embeddings = self.embedding_provider.embed(texts)
#             for product, embedding in zip(uncached, embeddings):
#                 self._product_cache[str(product.id)] = embedding

#         # Query is the only OpenAI call on every request
#         query_embedding = np.array(self.embedding_provider.embed([query])[0])
#         product_embeddings = np.array([self._product_cache[str(p.id)] for p in products])

#         scores = self._cosine_similarity(query_embedding, product_embeddings)

#         ranked = sorted(
#             zip(products, scores.tolist()),
#             key=lambda x: x[1],
#             reverse=True
#         )
#         return ranked

#     def _cosine_similarity(self, query_vec: np.ndarray, product_vecs: np.ndarray) -> np.ndarray:
#         query_norm = query_vec / np.linalg.norm(query_vec)
#         product_norms = product_vecs / np.linalg.norm(product_vecs, axis=1, keepdims=True)
#         return product_norms @ query_norm































#############################################################################################


# class EmbeddingMatcher(BaseMatcher):

#     def __init__(self, embedding_provider: OpenAIEmbeddingProvider):
#         self.embedding_provider = embedding_provider
#         self.cache = redis.from_url(redis_url, decode_responses=True)
#         self.ttl = 86400  # 24 hours
#     def clear_all_cache(self):
#         """Deletes all query and product embeddings from Redis."""
#         count = 0
#         # Scan for keys starting with our prefixes
#         for prefix in ["q:*", "p:*"]:
#             # scan_iter is efficient for production
#             for key in self.cache.scan_iter(prefix):
#                 self.cache.delete(key)
#                 count += 1
#         print(f"Successfully cleared {count} cached embeddings.")
#         return count

#     def rank(self, query: str, products: List[Product]) -> List[Tuple[Product, float]]:

#         # cahing the query in redis 
#         q_key = self.embedding_provider.generate_key(query, prefix="q")
#         cached_q = self.cache.get(q_key)
#         if cached_q:
#             query_embedding = json.loads(cached_q)
#         else:
#             # create query embedding
#             query_embedding = self.embedding_provider.embed([query])[0]
#             self.cache.setex(q_key, self.ttl, json.dumps(query_embedding))

#         # caching the products in redis

#         product_embs = []
#         to_embed_indices = []
#         to_embed_texts = []

#         for i, p in enumerate(products):
#             # We hash the product content (title + tags) to detect changes
#             p_text = f"{p.title} {' '.join(p.tags)}"
#             p_key = self.embedding_provider.generate_key(p_text, prefix="p")
            
#             cached_p = self.cache.get(p_key)
#             if cached_p:
#                 product_embs.append(json.loads(cached_p))
#             else:
#                 product_embs.append(None) # Placeholder
#                 to_embed_indices.append(i)
#                 to_embed_texts.append(p_text)

#         # 3. Embed only what is missing from cache
#         if to_embed_texts:
#             new_embs = self.embedding_provider.embed(to_embed_texts)
#             for i, emb in zip(to_embed_indices, new_embs):
#                 product_embs[i] = emb
#                 # Store in cache for future requests
#                 p_text = f"{products[i].title} {' '.join(products[i].tags)}"
#                 p_key = self.embedding_provider.generate_key(p_text, prefix="p")
#                 self.cache.setex(p_key, self.ttl, json.dumps(emb))

#         #create product mebeddings
#         # product_texts = [product_to_text(p) for p in products]
#         # product_embeddings = self.embedding_provider.embed(product_texts)

#         scores = self.cosine_similarity(query_embedding, product_embs)
#         ranked = sorted(zip(products, scores), key=lambda x: x[1], reverse=True)
#         return ranked

#         # scores = []
#         # # loop to calculate cosine similarity between query embedding and each product embedding
#         # for product, emb in zip(products, product_embeddings):
#         #     score = self.cosine_similarity(query_embedding, emb)
#         #     scores.append((product, score))

#         # #reverse list of scores based on score
#         # scores.sort(key=lambda x: x[1], reverse=True)
#         # return scores
    
#     def cosine_similarity(self, q_emb: List[float], p_embs: List[List[float]]) -> np.ndarray:
#         q = np.array(q_emb)
#         p = np.array(p_embs)
#         return np.dot(p, q) / (np.linalg.norm(q) * np.linalg.norm(p, axis=1) + 1e-9)
