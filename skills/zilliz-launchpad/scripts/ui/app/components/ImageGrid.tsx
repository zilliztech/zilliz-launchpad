import type { Hit } from "@/lib/milvus-client";

interface ImageGridProps {
  hits: Hit[];
  loading: boolean;
  error: string | null;
  hasSearched: boolean;
}

export function ImageGrid({ hits, loading, error, hasSearched }: ImageGridProps) {
  if (loading) {
    return <p style={{ color: "#888", marginTop: "1.5rem" }}>Searching…</p>;
  }
  if (error) {
    return (
      <p style={{ color: "#f87171", marginTop: "1.5rem" }}>Error: {error}</p>
    );
  }
  if (!hasSearched) {
    return (
      <p style={{ color: "#666", marginTop: "1.5rem" }}>
        Type a description above to search your photo collection.
      </p>
    );
  }
  if (hits.length === 0) {
    return (
      <p style={{ color: "#666", marginTop: "1.5rem" }}>
        No results — try a different query.
      </p>
    );
  }
  return (
    <section
      style={{
        marginTop: "1.5rem",
        display: "grid",
        gridTemplateColumns: "repeat(auto-fill, minmax(200px, 1fr))",
        gap: "1rem",
      }}
    >
      {hits.map((h) => (
        <ImageCard key={h.id} hit={h} />
      ))}
    </section>
  );
}

function ImageCard({ hit }: { hit: Hit }) {
  const { thumbnail_b64, width, height, bytes, taken_at } = hit.fields;
  const filename = hit.id.split("/").pop() ?? hit.id;
  return (
    <article
      style={{
        background: "#0f0f0f",
        border: "1px solid #222",
        borderRadius: 8,
        overflow: "hidden",
        display: "flex",
        flexDirection: "column",
      }}
    >
      <div
        style={{
          width: "100%",
          aspectRatio: "1 / 1",
          background: "#1a1a1a",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        {thumbnail_b64 ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={`data:image/jpeg;base64,${thumbnail_b64}`}
            alt={filename}
            style={{ width: "100%", height: "100%", objectFit: "cover" }}
          />
        ) : (
          <span style={{ color: "#666", fontSize: 12 }}>(no thumbnail)</span>
        )}
      </div>
      <div style={{ padding: "0.5rem 0.75rem", fontSize: 12, color: "#bbb" }}>
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            color: "#888",
            marginBottom: 4,
          }}
        >
          <span title={hit.id} style={{
            overflow: "hidden",
            textOverflow: "ellipsis",
            whiteSpace: "nowrap",
            maxWidth: "70%",
          }}>
            {filename}
          </span>
          <span>{hit.score.toFixed(3)}</span>
        </div>
        {(width || height) && (
          <div>
            {width ?? "?"}×{height ?? "?"}
            {typeof bytes === "number" && ` · ${formatBytes(bytes)}`}
          </div>
        )}
        {taken_at && (
          <div style={{ color: "#666", marginTop: 2 }}>{String(taken_at)}</div>
        )}
      </div>
    </article>
  );
}

function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(0)} KB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
}
