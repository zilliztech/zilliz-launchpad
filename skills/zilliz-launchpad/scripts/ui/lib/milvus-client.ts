export type SearchMode = "dense" | "sparse" | "hybrid";

export interface SearchRequest {
  query: string;
  top_k?: number;
  mode?: SearchMode;
}

export interface Hit {
  id: string;
  score: number;
  fields: Record<string, unknown>;
}

export interface SearchResponse {
  mode: SearchMode;
  hits: Hit[];
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
