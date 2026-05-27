# Milvus Client

Browser-based Milvus explorer for collection browsing, expr testing, dense vector search, optional BM25 hybrid search, and rerank result inspection.

## Run

```powershell
docker build -t milvus-client .
docker run --rm -p 8000:8000 milvus-client
```

Open `http://localhost:8000`.

## Local Test Stack

This starts Milvus standalone, the app, and a seeded `sample_documents` collection with dense vectors and a BM25 sparse field.

```powershell
docker compose -f docker-compose.test.yml up --build
```

Open `http://localhost:8000`, then use:

- Milvus host: `milvus`
- Milvus port: `19530`
- Embedding endpoint: `http://127.0.0.1:8000/api/mock/embedding`
- Rerank endpoint: `http://127.0.0.1:8000/api/mock/rerank`
- Collection: `sample_documents`
- Dense field: `dense`
- BM25 field: `sparse`

Try query text like `milvus bm25 hybrid search` and expr filters like `category == "milvus"`.

Stop the stack with:

```powershell
docker compose -f docker-compose.test.yml down
```

## Notes

- The app is stateless. Milvus, embedding, and rerank settings are stored in the browser localStorage.
- Query text is embedded through the configured embedding endpoint, then used for vector search.
- If a selected collection has a sparse vector field produced by Milvus BM25, the app can run dense + BM25 hybrid search.
- Expr filters are passed to Milvus search/query as scalar filters.
