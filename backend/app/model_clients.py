from typing import Any

import httpx

from .models import EmbeddingConfig, RerankConfig


def read_path(payload: Any, path: str) -> Any:
    current = payload
    for part in path.split("."):
        if isinstance(current, list):
            current = current[int(part)]
        else:
            current = current[part]
    return current


async def embed_text(text: str, config: EmbeddingConfig) -> list[float]:
    payload = await call_embedding(text, config)
    vector = read_path(payload, config.vector_path)
    if not isinstance(vector, list):
        raise ValueError(f"Embedding response path '{config.vector_path}' did not return a list")
    return [float(value) for value in vector]


async def call_embedding(text: str, config: EmbeddingConfig) -> Any:
    body = dict(config.extra_body)
    body[config.input_path] = text
    if config.model:
        body["model"] = config.model
    headers = {"Content-Type": "application/json"}
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(config.endpoint, json=body, headers=headers)
        response.raise_for_status()
        return response.json()


def find_embedding_paths(payload: Any) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []

    def walk(value: Any, path: str) -> None:
        if is_numeric_vector(value):
            candidates.append(
                {
                    "path": path,
                    "dimension": len(value),
                    "preview": [float(item) for item in value[:8]],
                }
            )
            return
        if isinstance(value, dict):
            for key, child in value.items():
                next_path = key if not path else f"{path}.{key}"
                walk(child, next_path)
        elif isinstance(value, list):
            for index, child in enumerate(value[:5]):
                next_path = str(index) if not path else f"{path}.{index}"
                walk(child, next_path)

    walk(payload, "")
    return sorted(candidates, key=lambda item: score_embedding_path(item["path"]), reverse=True)


def is_numeric_vector(value: Any) -> bool:
    if not isinstance(value, list) or len(value) < 2:
        return False
    sample = value[: min(len(value), 16)]
    return all(isinstance(item, int | float) and not isinstance(item, bool) for item in sample)


def score_embedding_path(path: str) -> int:
    lowered = path.lower()
    score = 0
    if "embedding" in lowered:
        score += 20
    if "vector" in lowered:
        score += 10
    if lowered.endswith("embedding"):
        score += 5
    return score


def document_text(hit: dict[str, Any]) -> str:
    entity = hit.get("entity") or {}
    if not isinstance(entity, dict):
        return str(entity)
    for value in entity.values():
        if isinstance(value, str) and value.strip():
            return value
    return str(entity)


async def rerank_hits(query: str, hits: list[dict[str, Any]], config: RerankConfig) -> list[dict[str, Any]]:
    documents = [document_text(hit) for hit in hits]
    body = dict(config.extra_body)
    body[config.query_path] = query
    body[config.documents_path] = documents
    if config.model:
        body["model"] = config.model
    headers = {"Content-Type": "application/json"}
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"
    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(config.endpoint, json=body, headers=headers)
        response.raise_for_status()
        payload = response.json()

    rerank_results = read_path(payload, config.results_path)
    enriched = [dict(hit, rerank_score=None, original_rank=index + 1) for index, hit in enumerate(hits)]
    for item in rerank_results:
        index = int(read_path(item, config.index_path))
        score = float(read_path(item, config.score_path))
        if 0 <= index < len(enriched):
            enriched[index]["rerank_score"] = score
    return sorted(enriched, key=lambda item: item["rerank_score"] if item["rerank_score"] is not None else -1, reverse=True)
