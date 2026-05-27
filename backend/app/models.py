from typing import Any, Literal

from pydantic import BaseModel, Field


class MilvusConnection(BaseModel):
    host: str
    port: int = 19530
    secure: bool = False
    token: str | None = None
    user: str | None = None
    password: str | None = None
    db_name: str | None = None


class BrowseRequest(BaseModel):
    connection: MilvusConnection
    collection_name: str
    expr: str = ""
    limit: int = Field(default=50, ge=1, le=500)
    offset: int = Field(default=0, ge=0, le=16000)
    output_fields: list[str] | None = None


class SearchParams(BaseModel):
    metric_type: str = "COSINE"
    params: dict[str, Any] = Field(default_factory=dict)


class EmbeddingConfig(BaseModel):
    endpoint: str
    model: str | None = None
    api_key: str | None = None
    input_path: str = "input"
    vector_path: str = "data.0.embedding"
    extra_body: dict[str, Any] = Field(default_factory=dict)


class RerankConfig(BaseModel):
    endpoint: str
    model: str | None = None
    api_key: str | None = None
    query_path: str = "query"
    documents_path: str = "documents"
    results_path: str = "results"
    index_path: str = "index"
    score_path: str = "relevance_score"
    extra_body: dict[str, Any] = Field(default_factory=dict)


class SearchRequest(BaseModel):
    connection: MilvusConnection
    collection_name: str
    query_text: str
    embedding: EmbeddingConfig
    rerank: RerankConfig | None = None
    dense_field: str
    bm25_field: str | None = None
    use_bm25: bool = True
    expr: str = ""
    limit: int = Field(default=10, ge=1, le=100)
    candidate_limit: int = Field(default=30, ge=1, le=500)
    output_fields: list[str] | None = None
    search_params: SearchParams = Field(default_factory=SearchParams)
    ranker: Literal["rrf", "weighted"] = "rrf"
    dense_weight: float = Field(default=0.7, ge=0.0, le=1.0)
    bm25_weight: float = Field(default=0.3, ge=0.0, le=1.0)


class ModelTestRequest(BaseModel):
    text: str
    embedding: EmbeddingConfig | None = None
    rerank: RerankConfig | None = None


class EmbeddingDetectRequest(BaseModel):
    text: str = "milvus vector search"
    embedding: EmbeddingConfig
