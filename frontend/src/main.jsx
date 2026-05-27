import React, { useEffect, useMemo, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { Database, Play, RefreshCcw, Search, Settings, SlidersHorizontal } from 'lucide-react';
import './styles.css';

const emptyConnection = { host: 'localhost', port: 19530, secure: false, token: '', user: '', password: '', db_name: '' };
const emptyEmbedding = { endpoint: '', model: '', api_key: '', input_path: 'input', vector_path: 'data.0.embedding', extra_body: {} };
const emptyRerank = { endpoint: '', model: '', api_key: '', query_path: 'query', documents_path: 'documents', results_path: 'results', index_path: 'index', score_path: 'relevance_score', extra_body: {} };

function loadState(key, fallback) {
  try {
    return JSON.parse(localStorage.getItem(key)) || fallback;
  } catch {
    return fallback;
  }
}

async function api(path, body) {
  const response = await fetch(`/api${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  const payload = await response.json();
  if (!response.ok) throw new Error(payload.detail || 'Request failed');
  return payload;
}

function Field({ label, children }) {
  return (
    <label className="field">
      <span>{label}</span>
      {children}
    </label>
  );
}

function App() {
  const [connection, setConnection] = useState(() => loadState('milvus.connection', emptyConnection));
  const [embedding, setEmbedding] = useState(() => loadState('milvus.embedding', emptyEmbedding));
  const [rerank, setRerank] = useState(() => loadState('milvus.rerank', emptyRerank));
  const [collections, setCollections] = useState([]);
  const [collectionFilter, setCollectionFilter] = useState('');
  const [selected, setSelected] = useState('');
  const [schema, setSchema] = useState(null);
  const [rows, setRows] = useState([]);
  const [expr, setExpr] = useState('');
  const [queryText, setQueryText] = useState('');
  const [denseField, setDenseField] = useState('');
  const [bm25Field, setBm25Field] = useState('');
  const [useBm25, setUseBm25] = useState(true);
  const [results, setResults] = useState(null);
  const [message, setMessage] = useState('');
  const [embeddingCandidates, setEmbeddingCandidates] = useState([]);

  useEffect(() => localStorage.setItem('milvus.connection', JSON.stringify(connection)), [connection]);
  useEffect(() => localStorage.setItem('milvus.embedding', JSON.stringify(embedding)), [embedding]);
  useEffect(() => localStorage.setItem('milvus.rerank', JSON.stringify(rerank)), [rerank]);

  const filteredCollections = useMemo(
    () => collections.filter((name) => name.toLowerCase().includes(collectionFilter.toLowerCase())),
    [collections, collectionFilter],
  );

  async function connect() {
    try {
      setMessage('Connecting...');
      const payload = await api('/connect', sanitizeConnection(connection));
      setCollections(payload.collections || []);
      setMessage(`Connected. ${payload.collections.length} collections found.`);
    } catch (error) {
      setMessage(error.message);
    }
  }

  async function selectCollection(name) {
    try {
      setSelected(name);
      setRows([]);
      setResults(null);
      const nextSchema = await api(`/collections/${name}/schema`, sanitizeConnection(connection));
      setSchema(nextSchema);
      setDenseField(nextSchema.dense_fields?.[0] || '');
      setBm25Field(nextSchema.sparse_fields?.[0] || '');
      setMessage(`${name} schema loaded.`);
    } catch (error) {
      setMessage(error.message);
    }
  }

  async function browse() {
    try {
      const payload = await api('/browse', {
        connection: sanitizeConnection(connection),
        collection_name: selected,
        expr,
        limit: 50,
        offset: 0,
      });
      setRows(payload.rows || []);
      setMessage(`Loaded ${payload.rows.length} rows.`);
    } catch (error) {
      setMessage(error.message);
    }
  }

  async function runSearch() {
    try {
      setMessage('Searching...');
      const payload = await api('/search', {
        connection: sanitizeConnection(connection),
        collection_name: selected,
        query_text: queryText,
        embedding: sanitizeModel(embedding),
        rerank: rerank.endpoint ? sanitizeModel(rerank) : null,
        dense_field: denseField,
        bm25_field: bm25Field || null,
        use_bm25: useBm25,
        expr,
        limit: 10,
        candidate_limit: 30,
        search_params: { metric_type: 'COSINE', params: {} },
        ranker: 'rrf',
      });
      setResults(payload);
      setMessage(`Search complete. ${payload.hits.length} hits.`);
    } catch (error) {
      setMessage(error.message);
    }
  }

  async function detectEmbeddingPath() {
    try {
      setMessage('Detecting embedding path...');
      const payload = await api('/embedding-detect', {
        text: queryText || 'milvus vector search',
        embedding: sanitizeModel(embedding),
      });
      setEmbeddingCandidates(payload.candidates || []);
      if (payload.selected) {
        setEmbedding({ ...embedding, vector_path: payload.selected });
        setMessage(`Embedding path detected: ${payload.selected}`);
      } else {
        setMessage('No numeric vector was found in the embedding response.');
      }
    } catch (error) {
      setMessage(error.message);
    }
  }

  return (
    <main>
      <aside>
        <div className="brand"><Database size={22} /> Milvus Client</div>
        <section>
          <h2>Connection</h2>
          <Field label="Host"><input value={connection.host} onChange={(e) => setConnection({ ...connection, host: e.target.value })} /></Field>
          <div className="grid2">
            <Field label="Port"><input type="number" value={connection.port} onChange={(e) => setConnection({ ...connection, port: Number(e.target.value) })} /></Field>
            <Field label="DB"><input value={connection.db_name || ''} onChange={(e) => setConnection({ ...connection, db_name: e.target.value })} /></Field>
          </div>
          <Field label="Token"><input type="password" value={connection.token || ''} onChange={(e) => setConnection({ ...connection, token: e.target.value })} /></Field>
          <label className="check"><input type="checkbox" checked={connection.secure} onChange={(e) => setConnection({ ...connection, secure: e.target.checked })} /> TLS</label>
          <button onClick={connect}><RefreshCcw size={16} /> Connect</button>
        </section>
        <section>
          <h2>Collections</h2>
          <input placeholder="Filter collections" value={collectionFilter} onChange={(e) => setCollectionFilter(e.target.value)} />
          <div className="collectionList">
            {filteredCollections.map((name) => (
              <button className={name === selected ? 'selected' : ''} key={name} onClick={() => selectCollection(name)}>{name}</button>
            ))}
          </div>
        </section>
      </aside>

      <div className="workspace">
        <header>
          <div>
            <h1>{selected || 'No collection selected'}</h1>
            <p>{message || 'Connect to Milvus and choose a collection.'}</p>
          </div>
          <button disabled={!selected} onClick={browse}><Play size={16} /> Browse</button>
        </header>

        <section className="panel">
          <h2><Settings size={18} /> Models</h2>
          <div className="modelGrid">
            <Field label="Embedding endpoint"><input value={embedding.endpoint} onChange={(e) => setEmbedding({ ...embedding, endpoint: e.target.value })} /></Field>
            <Field label="Embedding model"><input value={embedding.model} onChange={(e) => setEmbedding({ ...embedding, model: e.target.value })} /></Field>
            <Field label="Embedding API key"><input type="password" value={embedding.api_key} onChange={(e) => setEmbedding({ ...embedding, api_key: e.target.value })} /></Field>
            <Field label="Vector response path">
              <div className="inputAction">
                <input value={embedding.vector_path} onChange={(e) => setEmbedding({ ...embedding, vector_path: e.target.value })} />
                <button disabled={!embedding.endpoint} onClick={detectEmbeddingPath}>Detect</button>
              </div>
              {embeddingCandidates.length > 1 && (
                <select value={embedding.vector_path} onChange={(e) => setEmbedding({ ...embedding, vector_path: e.target.value })}>
                  {embeddingCandidates.map((candidate) => (
                    <option key={candidate.path} value={candidate.path}>{candidate.path} · dim {candidate.dimension}</option>
                  ))}
                </select>
              )}
            </Field>
            <Field label="Rerank endpoint"><input value={rerank.endpoint} onChange={(e) => setRerank({ ...rerank, endpoint: e.target.value })} /></Field>
            <Field label="Rerank model"><input value={rerank.model} onChange={(e) => setRerank({ ...rerank, model: e.target.value })} /></Field>
          </div>
        </section>

        <section className={`panel ${schema ? '' : 'mutedPanel'}`}>
          <h2><SlidersHorizontal size={18} /> Search</h2>
          <div className="searchGrid">
            <Field label="Query text"><textarea value={queryText} onChange={(e) => setQueryText(e.target.value)} /></Field>
            <Field label="Expr filter"><textarea value={expr} onChange={(e) => setExpr(e.target.value)} placeholder='category == "docs"' /></Field>
            <Field label="Dense field">
              <select value={denseField} disabled={!schema} onChange={(e) => setDenseField(e.target.value)}>
                {!schema && <option>Select a collection first</option>}
                {schema?.dense_fields.map((field) => <option key={field}>{field}</option>)}
              </select>
            </Field>
            <Field label="BM25 field">
              <select value={bm25Field} disabled={!schema} onChange={(e) => setBm25Field(e.target.value)}>
                <option value="">{schema ? 'None' : 'Select a collection first'}</option>
                {schema?.sparse_fields.map((field) => <option key={field}>{field}</option>)}
              </select>
            </Field>
          </div>
          <label className="check"><input type="checkbox" checked={useBm25} disabled={!schema} onChange={(e) => setUseBm25(e.target.checked)} /> Use BM25 when sparse field exists</label>
          <button disabled={!selected || !queryText || !embedding.endpoint || !denseField} onClick={runSearch}><Search size={16} /> Search</button>
        </section>

        <section className="split">
          <DataPanel title="Rows" data={rows} />
          <DataPanel title="Results" data={results?.reranked_hits || results?.hits || []} meta={results ? `${results.search_type} · BM25 ${results.bm25_available ? 'available' : 'unavailable'}` : ''} />
        </section>
      </div>
    </main>
  );
}

function sanitizeConnection(value) {
  return Object.fromEntries(Object.entries(value).filter(([, v]) => v !== ''));
}

function sanitizeModel(value) {
  return { ...Object.fromEntries(Object.entries(value).filter(([, v]) => v !== '')), extra_body: value.extra_body || {} };
}

function DataPanel({ title, data, meta }) {
  return (
    <section className="panel dataPanel">
      <h2>{title}</h2>
      {meta && <p className="meta">{meta}</p>}
      <pre>{JSON.stringify(data, null, 2)}</pre>
    </section>
  );
}

createRoot(document.getElementById('root')).render(<App />);
