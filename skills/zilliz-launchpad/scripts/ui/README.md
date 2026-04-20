# zilliz-launchpad demo UI

Next.js 16 + TypeScript. Talks to the FastAPI sidecar started by `zilliz_ops.py execute`.

Requires Node 20.9+.

```bash
pnpm install
pnpm dev             # http://localhost:3000
```

Override the sidecar URL via `NEXT_PUBLIC_SIDECAR_URL` at build time if your sidecar is not on `127.0.0.1:8000`.

No secrets go into the browser bundle — all credentials live in the sidecar process.
