from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()

# Initialize the embeddings model
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",  # or "text-embedding-3-large"
)

# --- 1. Embed a single text ---
text = "LangChain makes working with LLMs easy."
single_embedding = embeddings.embed_query(text)
print(f"Single embedding dimension: {len(single_embedding)}")
print(f"First 5 values: {single_embedding[:5]}")

# --- 2. Embed multiple documents ---
documents = [
    "LangChain is a framework for building LLM applications.",
    "OpenAI provides powerful embedding models.",
    "Embeddings represent text as dense vectors.",
]
doc_embeddings = embeddings.embed_documents(documents)
print(f"\nNumber of document embeddings: {len(doc_embeddings)}")
print(f"Each embedding dimension: {len(doc_embeddings[0])}")

# --- 3. Similarity search using cosine similarity ---
import numpy as np

def cosine_similarity(vec1, vec2):
    vec1, vec2 = np.array(vec1), np.array(vec2)
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

query = "What is LangChain?"
query_embedding = embeddings.embed_query(query)

print(f"\nQuery: '{query}'")
print("Similarity scores:")
for doc, doc_emb in zip(documents, doc_embeddings):
    score = cosine_similarity(query_embedding, doc_emb)
    print(f"  [{score:.4f}] {doc}")