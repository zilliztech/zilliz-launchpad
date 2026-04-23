export type SearchMode = "dense" | "sparse" | "hybrid";
export type Modality = "text" | "image";

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
