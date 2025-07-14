from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import httpx
import asyncio

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# URLs of your TF-IDF and Embedding search services
TFIDF_API_URL = "http://localhost:8001/search"
EMBEDDING_API_URL = "http://localhost:8002/search"

# Response schemas
class Document(BaseModel):
    doc_id: str
    text: str
    score: float

class SearchResponse(BaseModel):
    results: List[Document]

# Result merger
def merge_results(tfidf_results, embedding_results, alpha=0.9):
    merged = {}

    for doc in tfidf_results:
        merged[doc['doc_id']] = {
            "text": doc['text'],
            "score": alpha * doc['score']
        }

    for doc in embedding_results:
        if doc['doc_id'] in merged:
            merged[doc['doc_id']]['score'] += (1 - alpha) * doc['score']
        else:
            merged[doc['doc_id']] = {
                "text": doc['text'],
                "score": (1 - alpha) * doc['score']
            }

    combined = [
        {"doc_id": doc_id, "text": data["text"], "score": data["score"]}
        for doc_id, data in merged.items()
    ]
    return sorted(combined, key=lambda x: x["score"], reverse=True)

# ✅ Main hybrid endpoint
@app.get("/hybrid_search")
async def hybrid_search(query: str = Query(...)):
    async with httpx.AsyncClient() as client:
        tfidf_response = await client.post(TFIDF_API_URL, data={"query": query, "algorithm": "tfidf"})
        embed_response = await client.post(EMBEDDING_API_URL,data={"query": query, "algorithm": "embedding"})
        tfidf_results = tfidf_response.json().get("results", [])
        embedding_results = embed_response.json().get("results", [])
        combined_results = merge_results(tfidf_results, embedding_results)
        return {"results": combined_results}
        # return embed_response.json()

        # tfidf_response, embed_response = await asyncio.gather(tfidf_task, embed_task)
        # tfidf_response = await asyncio.gather(tfidf_task)


    # combined_results = merge_results(tfidf_results, embedding_results)

    # return {"results": combined_results}
    # return { "results": [{"doc_id": 1, "text": "This is a sample document.", "score": 0.8},{"doc_id": 2, "text": "Another document.", "score": 0.6}] };










