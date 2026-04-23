"use client";

import { useEffect, useState } from "react";
import { fetchInfo, searchSidecar, SearchMode, type Hit, type InfoResponse } from "@/lib/milvus-client";
import { ImageGrid } from "./components/ImageGrid";

export default function HomePage() {
  const [info, setInfo] = useState<InfoResponse | null>(null);
  const [infoError, setInfoError] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [mode, setMode] = useState<SearchMode>("dense");
  const [topK, setTopK] = useState(5);
  const [hits, setHits] = useState<Hit[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [hasSearched, setHasSearched] = useState(false);

  useEffect(() => {
    fetchInfo()
      .then(setInfo)
      .catch((err) => setInfoError(err instanceof Error ? err.message : String(err)));
  }, []);

  const isImage = info?.modality === "image";

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const resp = await searchSidecar({ query, mode: isImage ? "dense" : mode, top_k: topK });
      setHits(resp.hits);
      setHasSearched(true);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }

  return (
    <main style={{ maxWidth: isImage ? 1100 : 800, margin: "2rem auto", padding: "0 1rem" }}>
      <h1 style={{ fontSize: "1.75rem" }}>zilliz-launchpad</h1>
      <p style={{ color: "#888" }}>
        {isImage
          ? `Image search — ${info?.collection_name} (${info?.embedding.provider}/${info?.embedding.model})`
          : "Demo search — dense / sparse / hybrid"}
      </p>
      {infoError && (
        <p style={{ color: "#f87171" }}>
          Could not reach sidecar /info: {infoError}. Falling back to text-mode UI.
        </p>
      )}

      <form
        onSubmit={handleSubmit}
        style={{ display: "flex", gap: "0.5rem", marginTop: "1rem" }}
      >
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder={isImage ? "Describe an image…" : "Ask something…"}
          style={{ flex: 1 }}
          required
        />
        {!isImage && (
          <select value={mode} onChange={(e) => setMode(e.target.value as SearchMode)}>
            <option value="dense">Dense</option>
            <option value="sparse">Sparse</option>
            <option value="hybrid">Hybrid</option>
          </select>
        )}
        <input
          type="number"
          min={1}
          max={50}
          value={topK}
          onChange={(e) => setTopK(Number(e.target.value))}
          style={{ width: 70 }}
        />
        <button type="submit" disabled={loading}>
          {loading ? "…" : "Search"}
        </button>
      </form>

      {isImage ? (
        <ImageGrid hits={hits} loading={loading} error={error} hasSearched={hasSearched} />
      ) : (
        <TextResults hits={hits} loading={loading} error={error} hasSearched={hasSearched} />
      )}
    </main>
  );
}

function TextResults({
  hits,
  loading,
  error,
  hasSearched,
}: {
  hits: Hit[];
  loading: boolean;
  error: string | null;
  hasSearched: boolean;
}) {
  if (loading) {
    return <p style={{ color: "#888", marginTop: "1.5rem" }}>Searching…</p>;
  }
  if (error) {
    return <p style={{ color: "#f87171", marginTop: "1rem" }}>Error: {error}</p>;
  }
  if (!hasSearched) {
    return (
      <p style={{ color: "#666", marginTop: "1rem" }}>
        No results yet — try a query above.
      </p>
    );
  }
  if (hits.length === 0) {
    return (
      <p style={{ color: "#666", marginTop: "1rem" }}>
        No results — try a different query.
      </p>
    );
  }
  return (
    <section
      style={{
        marginTop: "1.5rem",
        display: "flex",
        flexDirection: "column",
        gap: "0.75rem",
      }}
    >
      {hits.map((h) => (
        <article
          key={h.id}
          style={{
            padding: "1rem",
            border: "1px solid #222",
            borderRadius: 8,
            background: "#0f0f0f",
          }}
        >
          <header style={{ display: "flex", justifyContent: "space-between", color: "#888" }}>
            <span>{h.id}</span>
            <span>score: {h.score.toFixed(4)}</span>
          </header>
          <pre style={{ whiteSpace: "pre-wrap", marginTop: "0.5rem" }}>
            {String(h.fields?.text ?? h.fields?.body ?? JSON.stringify(h.fields))}
          </pre>
        </article>
      ))}
    </section>
  );
}
