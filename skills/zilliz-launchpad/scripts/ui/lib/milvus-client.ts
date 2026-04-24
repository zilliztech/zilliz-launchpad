export type SearchMode = "dense" | "sparse" | "hybrid";
export type Modality = "text" | "image" | "video";

export interface SearchRequest {
  query: string;
  top_k?: number;
  mode?: SearchMode;
}

export interface Hit {
  id: string;
  score: number;
  fields: Record<string, unknown> & {
    thumbnail_b64?: string;
    width?: number;
    height?: number;
    bytes?: number;
    taken_at?: string;
    video_path?: string;
    t_seconds?: number;
    video_url?: string | null;
    video_url_warning?: string | null;
    source_index?: number;
  };
}

export interface SearchResponse {
  mode: SearchMode;
  modality: Modality;
  hits: Hit[];
}

export interface InfoResponse {
  collection_name: string;
  modality: Modality;
  primary_key: string;
  vector_field: string;
  sparse_enabled: boolean;
  embedding: {
    provider: string;
    model: string;
    dim: number;
  };
  has_thumbnails: boolean;
  video_static_prefix?: string | null;
  data_shape?: string | null;
}

export interface VideoFramesRequest {
  video_path: string;
  top_k?: number;
}

const BASE = process.env.NEXT_PUBLIC_SIDECAR_URL ?? "http://127.0.0.1:8000";

export async function searchSidecar(req: SearchRequest): Promise<SearchResponse> {
  const resp = await fetch(`${BASE}/search`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!resp.ok) {
    const body = await resp.text();
    throw new Error(`Sidecar ${resp.status}: ${body}`);
  }
  return (await resp.json()) as SearchResponse;
}

export async function fetchInfo(): Promise<InfoResponse> {
  const resp = await fetch(`${BASE}/info`);
  if (!resp.ok) {
    const body = await resp.text();
    throw new Error(`Sidecar /info ${resp.status}: ${body}`);
  }
  return (await resp.json()) as InfoResponse;
}

export const ACCEPTED_IMAGE_MIME_TYPES = [
  "image/jpeg",
  "image/png",
  "image/webp",
  "image/gif",
] as const;

export async function searchSidecarImage(
  file: File,
  topK = 10,
): Promise<SearchResponse> {
  const form = new FormData();
  form.append("file", file);
  form.append("top_k", String(topK));
  const resp = await fetch(`${BASE}/search_image`, {
    method: "POST",
    body: form,
  });
  if (!resp.ok) {
    // FastAPI wraps typed errors under `.detail.message`; fall back to raw
    // text for transport-level failures (413 size cap, CORS, network).
    let message = `Sidecar /search_image ${resp.status}`;
    try {
      const body = (await resp.json()) as { detail?: { message?: string } };
      if (body?.detail?.message) {
        message = body.detail.message;
      }
    } catch {
      const text = await resp.text();
      if (text) {
        message = `${message}: ${text}`;
      }
    }
    throw new Error(message);
  }
  return (await resp.json()) as SearchResponse;
}

export async function fetchVideoFrames(
  req: VideoFramesRequest,
): Promise<SearchResponse> {
  const resp = await fetch(`${BASE}/video_frames`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!resp.ok) {
    let message = `Sidecar /video_frames ${resp.status}`;
    try {
      const body = (await resp.json()) as { detail?: { message?: string } };
      if (body?.detail?.message) {
        message = body.detail.message;
      }
    } catch {
      /* ignore */
    }
    throw new Error(message);
  }
  return (await resp.json()) as SearchResponse;
}

export function sidecarBaseUrl(): string {
  return BASE;
}
