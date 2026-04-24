"use client";

import { useMemo, useRef, useState } from "react";
import {
  fetchVideoFrames,
  sidecarBaseUrl,
  type Hit,
} from "@/lib/milvus-client";

interface VideoResultsProps {
  hits: Hit[];
  loading: boolean;
  error: string | null;
  hasSearched: boolean;
}

interface VideoCluster {
  videoPath: string;
  primary: Hit;
  secondaries: Hit[];
  videoUrl: string | null | undefined;
  warning: string | null | undefined;
}

function formatTimestamp(seconds: number): string {
  const s = Math.floor(seconds);
  const mm = Math.floor(s / 60);
  const ss = s % 60;
  return `${String(mm).padStart(2, "0")}:${String(ss).padStart(2, "0")}`;
}

function groupByVideo(hits: Hit[]): VideoCluster[] {
  const groups = new Map<string, VideoCluster>();
  for (const hit of hits) {
    const videoPath = String(hit.fields?.video_path ?? "");
    if (!videoPath) continue;
    const existing = groups.get(videoPath);
    if (!existing) {
      groups.set(videoPath, {
        videoPath,
        primary: hit,
        secondaries: [],
        videoUrl: hit.fields?.video_url ?? null,
        warning: hit.fields?.video_url_warning ?? null,
      });
    } else if (existing.secondaries.length < 4) {
      existing.secondaries.push(hit);
    }
  }
  return Array.from(groups.values());
}

export function VideoResults({
  hits,
  loading,
  error,
  hasSearched,
}: VideoResultsProps) {
  const clusters = useMemo(() => groupByVideo(hits), [hits]);

  if (loading) {
    return <p style={{ color: "#888", marginTop: "1.5rem" }}>Searching…</p>;
  }
  if (error) {
    return <p style={{ color: "#f87171", marginTop: "1.5rem" }}>Error: {error}</p>;
  }
  if (!hasSearched) {
    return (
      <p style={{ color: "#666", marginTop: "1.5rem" }}>
        Describe a scene to find matching frames across your video library.
      </p>
    );
  }
  if (clusters.length === 0) {
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
        gridTemplateColumns: "repeat(auto-fill, minmax(260px, 1fr))",
        gap: "1rem",
      }}
    >
      {clusters.map((cluster) => (
        <VideoCard key={cluster.videoPath} cluster={cluster} />
      ))}
    </section>
  );
}

function VideoCard({ cluster }: { cluster: VideoCluster }) {
  const [expanded, setExpanded] = useState(false);
  const [extras, setExtras] = useState<Hit[]>([]);
  const [extrasError, setExtrasError] = useState<string | null>(null);
  const [extrasLoading, setExtrasLoading] = useState(false);
  const [codecError, setCodecError] = useState(false);
  const videoRef = useRef<HTMLVideoElement | null>(null);

  const primary = cluster.primary;
  const t = Number(primary.fields?.t_seconds ?? 0);
  const filename = cluster.videoPath.split("/").pop() ?? cluster.videoPath;
  const thumbnail = primary.fields?.thumbnail_b64;
  const base = sidecarBaseUrl();
  const videoSrc = cluster.videoUrl ? `${base}${cluster.videoUrl}` : null;

  async function expand() {
    if (!expanded) {
      setExpanded(true);
      if (extras.length === 0 && !extrasLoading) {
        setExtrasLoading(true);
        setExtrasError(null);
        try {
          const resp = await fetchVideoFrames({
            video_path: cluster.videoPath,
            top_k: 6,
          });
          // Filter to avoid redisplaying the primary or existing secondaries.
          const seen = new Set<string>([
            primary.id,
            ...cluster.secondaries.map((h) => h.id),
          ]);
          setExtras(resp.hits.filter((h) => !seen.has(h.id)).slice(0, 4));
        } catch (err: unknown) {
          setExtrasError(err instanceof Error ? err.message : String(err));
        } finally {
          setExtrasLoading(false);
        }
      }
    }
  }

  function seekTo(seconds: number) {
    const el = videoRef.current;
    if (el) {
      el.currentTime = seconds;
      void el.play().catch(() => {
        /* ignored — user may need to click play */
      });
    }
  }

  const secondaryFrames = [...cluster.secondaries, ...extras].slice(0, 4);

  return (
    <article
      onClick={expand}
      style={{
        padding: "0.75rem",
        border: "1px solid #222",
        borderRadius: 8,
        background: "#0f0f0f",
        cursor: expanded ? "default" : "pointer",
      }}
    >
      <header
        style={{
          display: "flex",
          justifyContent: "space-between",
          color: "#aaa",
          fontSize: "0.8rem",
          marginBottom: "0.5rem",
        }}
      >
        <span title={cluster.videoPath}>
          {filename} — {formatTimestamp(t)}
        </span>
        <span>score: {primary.score.toFixed(4)}</span>
      </header>

      {!expanded && thumbnail && (
        // eslint-disable-next-line @next/next/no-img-element
        <img
          src={`data:image/jpeg;base64,${thumbnail}`}
          alt={`Frame at ${formatTimestamp(t)}`}
          style={{ width: "100%", borderRadius: 4 }}
        />
      )}

      {expanded && videoSrc && !codecError && (
        <video
          ref={videoRef}
          src={videoSrc}
          controls
          preload="metadata"
          style={{ width: "100%", borderRadius: 4, background: "#000" }}
          onLoadedMetadata={() => {
            if (videoRef.current) videoRef.current.currentTime = t;
          }}
          onError={() => setCodecError(true)}
        />
      )}

      {expanded && (!videoSrc || codecError) && (
        <div style={{ color: "#f87171", fontSize: "0.85rem" }}>
          {codecError
            ? "Browser could not play this video; download it below."
            : cluster.warning ?? "No inline playback available for this path."}
          {videoSrc && (
            <>
              {" "}
              <a href={videoSrc} target="_blank" rel="noreferrer">
                Open file
              </a>
            </>
          )}
        </div>
      )}

      {extrasLoading && (
        <p style={{ color: "#666", fontSize: "0.8rem" }}>Loading more frames…</p>
      )}
      {extrasError && (
        <p style={{ color: "#f87171", fontSize: "0.8rem" }}>{extrasError}</p>
      )}

      {secondaryFrames.length > 0 && (
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fill, minmax(48px, 1fr))",
            gap: 4,
            marginTop: "0.5rem",
          }}
        >
          {secondaryFrames.map((frame) => {
            const frameT = Number(frame.fields?.t_seconds ?? 0);
            const thumb = frame.fields?.thumbnail_b64;
            return (
              <button
                key={frame.id}
                type="button"
                onClick={(e) => {
                  e.stopPropagation();
                  if (expanded && !codecError && videoSrc) {
                    seekTo(frameT);
                  } else {
                    expand();
                  }
                }}
                title={`Seek to ${formatTimestamp(frameT)}`}
                style={{
                  padding: 0,
                  border: "1px solid #333",
                  background: "#111",
                  cursor: "pointer",
                }}
              >
                {thumb ? (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img
                    src={`data:image/jpeg;base64,${thumb}`}
                    alt={`frame at ${formatTimestamp(frameT)}`}
                    style={{ width: "100%", display: "block" }}
                  />
                ) : (
                  <span style={{ color: "#777", fontSize: "0.7rem" }}>
                    {formatTimestamp(frameT)}
                  </span>
                )}
              </button>
            );
          })}
        </div>
      )}
    </article>
  );
}
