from typing import Any

from pymilvus import AnnSearchRequest, MilvusClient, RRFRanker, WeightedRanker

from .models import MilvusConnection, SearchRequest


VECTOR_TYPES = {"FLOAT_VECTOR", "BINARY_VECTOR", "FLOAT16_VECTOR", "BFLOAT16_VECTOR", "SPARSE_FLOAT_VECTOR"}
SPARSE_TYPES = {"SPARSE_FLOAT_VECTOR"}


def get_client(connection: MilvusConnection) -> MilvusClient:
    scheme = "https" if connection.secure else "http"
    uri = f"{scheme}://{connection.host}:{connection.port}"
    kwargs: dict[str, Any] = {"uri": uri}
    if connection.token:
        kwargs["token"] = connection.token
    elif connection.user or connection.password:
        kwargs["user"] = connection.user
        kwargs["password"] = connection.password
    if connection.db_name:
        kwargs["db_name"] = connection.db_name
    return MilvusClient(**kwargs)


def normalize_field(field: dict[str, Any]) -> dict[str, Any]:
    raw_dtype = field.get("type", "")
    dtype_name = getattr(raw_dtype, "name", str(raw_dtype).split(".")[-1])
    return {
        "name": field.get("name"),
        "type": dtype_name,
        "is_primary": field.get("is_primary", False),
        "auto_id": field.get("auto_id", False),
        "description": field.get("description", ""),
        "params": field.get("params", {}),
        "is_vector": dtype_name in VECTOR_TYPES,
        "is_sparse": dtype_name in SPARSE_TYPES,
    }


def describe_collection(client: MilvusClient, collection_name: str) -> dict[str, Any]:
    desc = client.describe_collection(collection_name)
    fields = [normalize_field(field) for field in desc.get("fields", [])]
    return {
        "name": collection_name,
        "description": desc.get("description", ""),
        "auto_id": desc.get("auto_id", False),
        "fields": fields,
        "dense_fields": [field["name"] for field in fields if field["is_vector"] and not field["is_sparse"]],
        "sparse_fields": [field["name"] for field in fields if field["is_sparse"]],
        "scalar_fields": [field["name"] for field in fields if not field["is_vector"]],
    }


def preview_value(value: Any) -> Any:
    if isinstance(value, list) and len(str(value)) > 120:
        return f"{str(value)[:100]}... ({len(value)} values)"
    if isinstance(value, dict):
        return {key: preview_value(child) for key, child in value.items()}
    if isinstance(value, str) and len(value) > 500:
        return value[:500] + "..."
    return value


def preview_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{key: preview_value(value) for key, value in row.items()} for row in rows]


def hit_to_dict(hit: Any) -> dict[str, Any]:
    if isinstance(hit, dict):
        entity = hit.get("entity", {})
        result = {
            "id": hit.get("id"),
            "score": hit.get("distance"),
            "entity": preview_value(entity),
        }
        return result
    entity = getattr(hit, "entity", {}) or {}
    return {
        "id": getattr(hit, "id", None),
        "score": getattr(hit, "distance", None),
        "entity": preview_value(entity),
    }


def build_output_fields(schema: dict[str, Any], requested: list[str] | None) -> list[str]:
    if requested:
        return requested
    return schema["scalar_fields"] or ["*"]


def run_search(client: MilvusClient, request: SearchRequest, query_embedding: list[float]) -> dict[str, Any]:
    schema = describe_collection(client, request.collection_name)
    output_fields = build_output_fields(schema, request.output_fields)
    client.load_collection(request.collection_name)

    search_filter = request.expr or None
    bm25_available = bool(request.bm25_field and request.bm25_field in schema["sparse_fields"])
    use_hybrid = request.use_bm25 and bm25_available

    if use_hybrid:
        dense_req = AnnSearchRequest(
            data=[query_embedding],
            anns_field=request.dense_field,
            param={"metric_type": request.search_params.metric_type, "params": request.search_params.params},
            limit=request.candidate_limit,
            expr=search_filter,
        )
        bm25_req = AnnSearchRequest(
            data=[request.query_text],
            anns_field=request.bm25_field,
            param={"metric_type": "BM25", "params": {}},
            limit=request.candidate_limit,
            expr=search_filter,
        )
        ranker = RRFRanker() if request.ranker == "rrf" else WeightedRanker(request.dense_weight, request.bm25_weight)
        raw = client.hybrid_search(
            collection_name=request.collection_name,
            reqs=[dense_req, bm25_req],
            ranker=ranker,
            limit=request.limit,
            output_fields=output_fields,
        )
        search_type = "hybrid_dense_bm25"
    else:
        raw = client.search(
            collection_name=request.collection_name,
            data=[query_embedding],
            anns_field=request.dense_field,
            search_params={"metric_type": request.search_params.metric_type, "params": request.search_params.params},
            filter=search_filter,
            limit=request.limit,
            output_fields=output_fields,
        )
        search_type = "dense"

    hits = [hit_to_dict(hit) for hit in (raw[0] if raw else [])]
    return {"search_type": search_type, "bm25_available": bm25_available, "hits": hits}
