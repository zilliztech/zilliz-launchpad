"use client";

import { useEffect, useRef, useState } from "react";
import {
  ACCEPTED_IMAGE_MIME_TYPES,
  fetchInfo,
  searchSidecar,
  searchSidecarImage,
  SearchMode,
  type Hit,
  type InfoResponse,
} from "@/lib/milvus-client";
import { ImageGrid } from "./components/ImageGrid";
import { VideoResults } from "./components/VideoResults";

const ACCEPT_ATTR = ACCEPTED_IMAGE_MIME_TYPES.join(",");

export default function HomePage() {
  const [info, setInfo] = useState<InfoResponse | null>(null);
  const [infoError, setInfoError] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [mode, setMode] = useState<SearchMode>("dense");
  const [topK, setTopK] = useState(5);
  const [filter, setFilter] = useState("");
  const [rerank, setRerank] = useState<"off" | "default">("off");
  const [hits, setHits] = useState<Hit[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [lastQueryImage, setLastQueryImage] = useState<string | null>(null);
  const [hasSearched, setHasSearched] = useState(false);
  const [isDropping, setIsDropping] = useState(false);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    fetchInfo()
      .then(setInfo)
      .catch((err) => setInfoError(err instanceof Error ? err.message : String(err)));
  }, []);

  const isImage = info?.modality === "image";
  const isVideo = info?.modality === "video";
  const visual = isImage || isVideo;

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setLoading(true);
    setError(null);
    try {
      const resp = await searchSidecar({
        query,
        mode: visual ? "dense" : mode,
        top_k: topK,
        filter: !visual && filter ? filter : undefined,
        rerank: !visual && rerank !== "off" ? rerank : undefined,
      });
      setHits(resp.hits);
      setHasSearched(true);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }

  async function runImageSearch(file: File) {
    // Client-side MIME guard: the sidecar rejects decode failures with 400,
    // but catching non-images here avoids a pointless round-trip.
    const mimeOk = (ACCEPTED_IMAGE_MIME_TYPES as readonly string[]).includes(file.type);
    if (!mimeOk) {
      setUploadError(
        `Unsupported file type: ${file.type || "unknown"}. Supported: JPEG, PNG, WebP, GIF.`,
      );
      return;
    }
    setUploadError(null);
    setLoading(true);
    try {
      const resp = await searchSidecarImage(file, topK);
      setHits(resp.hits);
      setHasSearched(true);
      setLastQueryImage(file.name);
    } catch (err: unknown) {
      // Sidecar error: show inline, keep existing grid visible.
      setUploadError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }

  function handleFilePicked(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (file) {
      void runImageSearch(file);
    }
    // Clear the input so picking the same file twice still fires change.
    e.target.value = "";
  }

  function handleDrop(e: React.DragEvent<HTMLElement>) {
    e.preventDefault();
    setIsDropping(false);
    const file = e.dataTransfer.files?.[0];
    if (file) {
      void runImageSearch(file);
    }
  }

  function handleDragOver(e: React.DragEvent<HTMLElement>) {
    e.preventDefault();
    if (!isDropping) setIsDropping(true);
  }

  function handleDragLeave(e: React.DragEvent<HTMLElement>) {
    e.preventDefault();
    setIsDropping(false);
  }

  return (
    <main
      style={{ maxWidth: visual ? 1100 : 800, margin: "2rem auto", padding: "0 1rem" }}
      onDragOver={visual ? handleDragOver : undefined}
      onDragLeave={visual ? handleDragLeave : undefined}
      onDrop={visual ? handleDrop : undefined}
    >
      <h1 style={{ fontSize: "1.75rem" }}>zilliz-launchpad</h1>
      <p style={{ color: "#888" }}>
        {isVideo
          ? `Video search — ${info?.collection_name} (${info?.embedding.provider}/${info?.embedding.model})`
          : isImage
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
        style={{ display: "flex", gap: "0.5rem", marginTop: "1rem", flexWrap: "wrap" }}
      >
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder={
            isVideo
              ? "Describe a scene…"
              : isImage
                ? "Describe an image…"
                : "Ask something…"
          }
          style={{ flex: 1, minWidth: 200 }}
          required
        />
        {!visual && (
          <>
            <select value={mode} onChange={(e) => setMode(e.target.value as SearchMode)}>
              <option value="dense">Dense</option>
              <option value="sparse">Sparse</option>
              <option value="hybrid">Hybrid</option>
            </select>
            <select
              value={rerank}
              onChange={(e) => setRerank(e.target.value as "off" | "default")}
              title="Reranker"
            >
              <option value="off">Rerank: Off</option>
              <option value="default" disabled={!info?.default_reranker}>
                {info?.default_reranker
                  ? `Rerank: Default (${info.default_reranker})`
                  : "Rerank: Default (none configured)"}
              </option>
            </select>
          </>
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
        {!visual && (
          <input
            type="text"
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            placeholder='Filter expression (e.g. year >= 2023)'
            style={{ flexBasis: "100%", minWidth: 200 }}
          />
        )}
        {visual && (
          <>
            <input
              ref={fileInputRef}
              type="file"
              accept={ACCEPT_ATTR}
              onChange={handleFilePicked}
              style={{ display: "none" }}
              aria-label="Upload query image"
            />
            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              disabled={loading}
              title="Search by example image"
            >
              Search by image…
            </button>
          </>
        )}
      </form>

      {visual && (
        <div style={{ marginTop: "0.75rem", display: "flex", flexDirection: "column", gap: "0.25rem" }}>
          <p style={{ color: isDropping ? "#38bdf8" : "#555", fontSize: "0.85rem" }}>
            {isDropping ? "Drop image to search" : "Tip: drop an image anywhere on this page to search by example"}
          </p>
          {lastQueryImage && !uploadError && (
            <p style={{ color: "#888", fontSize: "0.85rem" }}>
              Last image query: <code>{lastQueryImage}</code>
            </p>
          )}
          {uploadError && (
            <p style={{ color: "#f87171", fontSize: "0.85rem" }}>{uploadError}</p>
          )}
        </div>
      )}

      {isVideo ? (
        <VideoResults hits={hits} loading={loading} error={error} hasSearched={hasSearched} />
      ) : isImage ? (
        <ImageGrid hits={hits} loading={loading} error={error} hasSearched={hasSearched} />
      ) : (
        <TextResults
          hits={hits}
          loading={loading}
          error={error}
          hasSearched={hasSearched}
          info={info}
        />
      )}
    </main>
  );
}

const TEXT_FIELD_CANDIDATES = ["text", "body", "content", "chunk"] as const;

function pickPrimaryText(fields: Hit["fields"]): { key: string | null; value: string | null } {
  for (const key of TEXT_FIELD_CANDIDATES) {
    const v = fields[key];
    if (typeof v === "string" && v.length > 0) return { key, value: v };
  }
  // Fallback: first string-valued field
  for (const [k, v] of Object.entries(fields)) {
    if (typeof v === "string" && v.length > 0) return { key: k, value: v };
  }
  return { key: null, value: null };
}

function TextResults({
  hits,
  loading,
  error,
  hasSearched,
  info,
}: {
  hits: Hit[];
  loading: boolean;
  error: string | null;
  hasSearched: boolean;
  info: InfoResponse | null;
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
      {hits.map((h) => {
        const fields = h.fields ?? {};
        const { key: textKey, value: textValue } = pickPrimaryText(fields);
        const pk = info?.primary_key;
        const metaEntries = Object.entries(fields).filter(([k, v]) => {
          if (k === textKey) return false;
          if (pk && k === pk) return false;
          if (v === null || v === undefined) return false;
          if (typeof v === "string" && v.length === 0) return false;
          if (Array.isArray(v) || typeof v === "object") return false;
          return true;
        });
        return (
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
            {metaEntries.length > 0 && (
              <div
                style={{
                  marginTop: "0.5rem",
                  display: "flex",
                  flexWrap: "wrap",
                  gap: "0.5rem",
                  color: "#9ca3af",
                  fontSize: "0.85rem",
                }}
              >
                {metaEntries.map(([k, v], i) => (
                  <span key={k}>
                    <span style={{ color: "#6b7280" }}>{k}:</span>{" "}
                    <span style={{ color: "#d1d5db" }}>{String(v)}</span>
                    {i < metaEntries.length - 1 && <span style={{ marginLeft: "0.5rem" }}>·</span>}
                  </span>
                ))}
              </div>
            )}
            {textValue !== null ? (
              <pre style={{ whiteSpace: "pre-wrap", marginTop: "0.5rem" }}>{textValue}</pre>
            ) : metaEntries.length === 0 ? (
              <pre style={{ whiteSpace: "pre-wrap", marginTop: "0.5rem" }}>
                {JSON.stringify(fields)}
              </pre>
            ) : null}
          </article>
        );
      })}
    </section>
  );
}
