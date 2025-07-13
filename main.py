from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import List, Dict
import httpx
import asyncio

app = FastAPI()

# Config
TFIDF_API_URL = "http://localhost:8001/search"
EMBEDDING_API_URL = "http://localhost:8002/search"

class Document(BaseModel):
    doc_id: str
    text: str
    score: float

class SearchResponse(BaseModel):
    results: List[Document]

def merge_results(tfidf_results, embedding_results, alpha=0.5):
    """
    Merge TF-IDF and embedding results using a weighted score.
    alpha: weight for TF-IDF, (1 - alpha) for embedding
    """
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

    # Convert to list and sort by score descending
    combined = [
        {"doc_id": doc_id, "text": data["text"], "score": data["score"]}
        for doc_id, data in merged.items()
    ]
    return sorted(combined, key=lambda x: x["score"], reverse=True)

@app.get("/hybrid_search", response_model=SearchResponse)
async def hybrid_search(query: str = Query(...)):
    async with httpx.AsyncClient() as client:
        tfidf_task = client.get(TFIDF_API_URL, params={"query": query})
        embed_task = client.get(EMBEDDING_API_URL, params={"query": query})

        tfidf_response, embed_response = await asyncio.gather(tfidf_task, embed_task)

    tfidf_results = tfidf_response.json()
    embedding_results = embed_response.json()

    combined_results = merge_results(tfidf_results, embedding_results)

    return {"results": combined_results}
