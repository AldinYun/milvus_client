from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from .milvus_service import describe_collection, get_client, preview_rows, run_search
from .model_clients import call_embedding, embed_text, find_embedding_paths, rerank_hits
from .models import BrowseRequest, EmbeddingDetectRequest, MilvusConnection, ModelTestRequest, SearchRequest


app = FastAPI(title="Milvus Client", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def raise_api_error(error: Exception) -> None:
    raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/connect")
def connect(connection: MilvusConnection) -> dict[str, object]:
    try:
        client = get_client(connection)
        collections = client.list_collections()
        return {"ok": True, "collections": collections}
    except Exception as error:
        raise_api_error(error)


@app.post("/api/collections")
def collections(connection: MilvusConnection) -> dict[str, object]:
    try:
        client = get_client(connection)
        names = client.list_collections()
        return {"collections": names}
    except Exception as error:
        raise_api_error(error)


@app.post("/api/collections/{collection_name}/schema")
def collection_schema(collection_name: str, connection: MilvusConnection) -> dict[str, object]:
    try:
        client = get_client(connection)
        return describe_collection(client, collection_name)
    except Exception as error:
        raise_api_error(error)


@app.post("/api/browse")
def browse(request: BrowseRequest) -> dict[str, object]:
    try:
        client = get_client(request.connection)
        schema = describe_collection(client, request.collection_name)
        output_fields = request.output_fields or schema["scalar_fields"] or ["*"]
        rows = client.query(
            collection_name=request.collection_name,
            filter=request.expr,
            output_fields=output_fields,
            limit=request.limit,
            offset=request.offset,
        )
        return {"rows": preview_rows(rows), "schema": schema}
    except Exception as error:
        raise_api_error(error)


@app.post("/api/search")
async def search(request: SearchRequest) -> dict[str, object]:
    try:
        vector = await embed_text(request.query_text, request.embedding)
        client = get_client(request.connection)
        result = run_search(client, request, vector)
        if request.rerank:
            result["reranked_hits"] = await rerank_hits(request.query_text, result["hits"], request.rerank)
        return result
    except Exception as error:
        raise_api_error(error)


@app.post("/api/model-test")
async def model_test(request: ModelTestRequest) -> dict[str, object]:
    try:
        result: dict[str, object] = {}
        if request.embedding:
            vector = await embed_text(request.text, request.embedding)
            result["embedding_dimension"] = len(vector)
            result["embedding_preview"] = vector[:8]
        if request.rerank:
            sample_hits = [{"entity": {"text": request.text}}, {"entity": {"text": "unrelated sample"}}]
            result["rerank_preview"] = await rerank_hits(request.text, sample_hits, request.rerank)
        return result
    except Exception as error:
        raise_api_error(error)


@app.post("/api/embedding-detect")
async def embedding_detect(request: EmbeddingDetectRequest) -> dict[str, object]:
    try:
        payload = await call_embedding(request.text, request.embedding)
        candidates = find_embedding_paths(payload)
        return {"candidates": candidates, "selected": candidates[0]["path"] if candidates else None}
    except Exception as error:
        raise_api_error(error)


@app.post("/api/mock/embedding")
async def mock_embedding(payload: dict[str, object]) -> dict[str, object]:
    text = str(payload.get("input", ""))
    vector = deterministic_embedding(text)
    return {"data": [{"embedding": vector}]}


@app.post("/api/mock/rerank")
async def mock_rerank(payload: dict[str, object]) -> dict[str, object]:
    query = str(payload.get("query", "")).lower()
    documents = payload.get("documents", [])
    if not isinstance(documents, list):
        documents = []
    query_terms = {term for term in query.replace("-", " ").split() if term}
    results = []
    for index, document in enumerate(documents):
        doc_terms = set(str(document).lower().replace("-", " ").split())
        overlap = len(query_terms & doc_terms)
        results.append({"index": index, "relevance_score": float(overlap) + (1.0 / (index + 1))})
    results.sort(key=lambda item: item["relevance_score"], reverse=True)
    return {"results": results}


def deterministic_embedding(text: str) -> list[float]:
    buckets = [0.0] * 8
    for index, char in enumerate(text.lower()):
        buckets[index % len(buckets)] += (ord(char) % 31) / 31.0
    norm = sum(value * value for value in buckets) ** 0.5 or 1.0
    return [round(value / norm, 6) for value in buckets]


static_dir = Path(__file__).resolve().parents[2] / "frontend" / "dist"
if static_dir.exists():
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="frontend")
