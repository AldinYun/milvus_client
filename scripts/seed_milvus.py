import math
import os
import time

from pymilvus import DataType, Function, FunctionType, MilvusClient


COLLECTION = os.getenv("MILVUS_COLLECTION", "sample_documents")
URI = os.getenv("MILVUS_URI", "http://milvus-standalone:19530")


DOCUMENTS = [
    {
        "id": 1,
        "title": "Milvus hybrid search",
        "text": "Milvus supports dense vector search and BM25 sparse full text search for retrieval augmented generation.",
        "category": "milvus",
        "source": "guide",
    },
    {
        "id": 2,
        "title": "Azure AI Search concepts",
        "text": "Azure AI Search combines keyword search, vector search, filters, scoring profiles, and semantic ranking.",
        "category": "azure",
        "source": "reference",
    },
    {
        "id": 3,
        "title": "Expr scalar filters",
        "text": "Milvus expr filters can restrict search results by category, source, numeric values, and other scalar fields.",
        "category": "milvus",
        "source": "guide",
    },
    {
        "id": 4,
        "title": "Reranking search results",
        "text": "A rerank model receives the query and candidate documents, then returns a new relevance score and ordering.",
        "category": "rerank",
        "source": "note",
    },
    {
        "id": 5,
        "title": "Embedding model setup",
        "text": "An embedding model transforms query text into a dense vector so Milvus can perform semantic search.",
        "category": "embedding",
        "source": "note",
    },
]


def deterministic_embedding(text: str) -> list[float]:
    buckets = [0.0] * 8
    for index, char in enumerate(text.lower()):
        buckets[index % len(buckets)] += (ord(char) % 31) / 31.0
    norm = math.sqrt(sum(value * value for value in buckets)) or 1.0
    return [round(value / norm, 6) for value in buckets]


def wait_for_milvus() -> MilvusClient:
    last_error = None
    for _ in range(60):
        try:
            client = MilvusClient(uri=URI)
            client.list_collections()
            return client
        except Exception as error:
            last_error = error
            time.sleep(2)
    raise RuntimeError(f"Milvus did not become ready: {last_error}")


def main() -> None:
    client = wait_for_milvus()
    if client.has_collection(COLLECTION):
        client.drop_collection(COLLECTION)

    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=256)
    schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=2048, enable_analyzer=True)
    schema.add_field(field_name="category", datatype=DataType.VARCHAR, max_length=64)
    schema.add_field(field_name="source", datatype=DataType.VARCHAR, max_length=64)
    schema.add_field(field_name="dense", datatype=DataType.FLOAT_VECTOR, dim=8)
    schema.add_field(field_name="sparse", datatype=DataType.SPARSE_FLOAT_VECTOR)

    schema.add_function(
        Function(
            name="text_bm25",
            input_field_names=["text"],
            output_field_names=["sparse"],
            function_type=FunctionType.BM25,
        )
    )

    index_params = client.prepare_index_params()
    index_params.add_index(field_name="dense", index_type="FLAT", metric_type="COSINE")
    index_params.add_index(field_name="sparse", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25")

    client.create_collection(collection_name=COLLECTION, schema=schema, index_params=index_params)
    rows = [{**doc, "dense": deterministic_embedding(doc["text"])} for doc in DOCUMENTS]
    client.insert(collection_name=COLLECTION, data=rows)
    client.flush(collection_name=COLLECTION)
    client.load_collection(collection_name=COLLECTION)
    print(f"Seeded {COLLECTION} with {len(rows)} documents at {URI}")


if __name__ == "__main__":
    main()

