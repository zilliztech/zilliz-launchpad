"use client";

import { useState } from "react";
import { searchSidecar, SearchMode, type Hit } from "@/lib/milvus-client";

export default function HomePage() {
  const [query, setQuery] = useState("");
  const [mode, setMode] = useState<SearchMode>("dense");
  const [topK, setTopK] = useState(5);
  const [hits, setHits] = useState<Hit[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const resp = await searchSidecar({ query, mode, top_k: topK });
      setHits(resp.hits);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }

  return (
    <main style={{ maxWidth: 800, margin: "2rem auto", padding: "0 1rem" }}>
      <h1 style={{ fontSize: "1.75rem" }}>zilliz-launchpad</h1>
      <p style={{ color: "#888" }}>Demo search — dense / sparse / hybrid</p>

      <form onSubmit={handleSubmit} style={{ display: "flex", gap: "0.5rem", marginTop: "1rem" }}>
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Ask something…"
          style={{ flex: 1 }}
          required
        />
        <select value={mode} onChange={(e) => setMode(e.target.value as SearchMode)}>
          <option value="dense">Dense</option>
          <option value="sparse">Sparse</option>
          <option value="hybrid">Hybrid</option>
        </select>
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

      {error && (
        <p style={{ marginTop: "1rem", color: "#f87171" }}>Error: {error}</p>
      )}

      <section style={{ marginTop: "1.5rem", display: "flex", flexDirection: "column", gap: "0.75rem" }}>
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
        {hits.length === 0 && !loading && !error && (
          <p style={{ color: "#666" }}>No results yet — try a query above.</p>
        )}
      </section>
    </main>
  );
}
