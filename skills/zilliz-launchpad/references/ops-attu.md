# Attu ops UI — playbooks

Attu is an optional, opt-in Milvus admin UI that ships with launchpad as a
developer/ops tool. It is **not** a replacement for the Next.js demo UI — the
demo UI is what you show end users; Attu is what you (or an ops engineer) use to
look under the hood.

Run it with:

```bash
cd skills/zilliz-launchpad/scripts
./start_milvus.sh attu up       # → http://localhost:8000
./start_milvus.sh attu status
./start_milvus.sh attu down
```

The service is declared under compose profile `ops`, so the default
`./start_milvus.sh up` never pulls or starts it. The web UI is bound to
`127.0.0.1:8000` only — it is not exposed to your LAN by design (Attu has no
built-in auth).

To connect Attu to a remote cluster (Zilliz Cloud, a staging Milvus, or a
BYOC deployment), set env vars before `attu up`:

```bash
export ATTU_MILVUS_URL=https://<cluster>.api.cloud.zilliz.com
export ATTU_MILVUS_TOKEN=<cluster-token>
./start_milvus.sh attu down && ./start_milvus.sh attu up
```

When either variable is unset, Attu falls back to `standalone:19530` over the
compose network.

## Playbook 1 — Post-Execute data verification

After Phase 4 (`zilliz_ops.py execute`) completes, use Attu to confirm the data
actually landed the way plan.json said it would:

1. `./start_milvus.sh attu up` and open `http://localhost:8000`.
2. In the left sidebar, pick the database, then the collection listed in
   `execute.json → collection_status`.
3. **Schema tab** — confirm the primary-key field, vector field name and dim,
   and every scalar field from `plan.json → schema.fields` are present.
4. **Data Preview tab** — sample 20 rows. Spot-check:
   - the primary key is unique and populated
   - vector fields are non-empty (`[0.0123, …, 512 dims]` — not `null`)
   - scalar fields match the source JSONL (e.g. titles not truncated, tags not
     collapsed to `None`)
5. **Partitions tab** (if plan used partitions) — confirm row counts per
   partition roughly match expectations; a wildly skewed split usually means
   the partition key wasn't routed correctly upstream.

If anything is off, re-run `execute` with `--no-ui` on a scratch run-dir rather
than mutating the current one.

## Playbook 2 — Evaluate bad-case drill-down

When `evaluate.md` flags a recall drop, Attu's Vector Search tab is the
fastest way to reproduce the query interactively:

1. Open the collection in Attu, go to **Vector Search**.
2. Paste the offending query vector (pull it from `eval_report.json → queries[i]
   .vector`) OR type the raw text if your collection has a text-in search
   function configured.
3. Set `topK` to the same value used in evaluate (default 10), and set `metric
   type` and `search params` (nprobe / ef) to the values from `plan.json →
   index`.
4. Inspect the top-K result — what actually came back? Common findings:
   - Expected doc is in top 50 but not top 10 → tune `ef` / `nprobe` upward and
     re-run evaluate
   - Expected doc has a stale field (old tag, wrong language) → ingest issue,
     not a retrieval issue
   - Expected doc is missing entirely → went missing during ingest
     (`execute.json → skipped_files`)
5. Add a filter expression in the same tab (e.g. `lang == "en"`) to reproduce
   hybrid-filter bad-cases.

Capture the finding as a bullet in the eval report and move on — Attu is the
drill-down tool, not the scoring tool.

## Playbook 3 — Post-Deploy Cloud ops

After Phase 6 promotes the collection to Zilliz Cloud, point Attu at the
Cloud endpoint to handle anything the launchpad CLI doesn't:

1. Export `ATTU_MILVUS_URL` and `ATTU_MILVUS_TOKEN` from `deploy.json →
   public_endpoint` and your `ZILLIZ_TOKEN`, then `./start_milvus.sh attu
   down && ./start_milvus.sh attu up`.
2. **Overview tab** — look at load state, segment count, and compaction status.
   A stuck "Loading" state is usually a quota or replica configuration issue
   visible here before anywhere else.
3. **Partitions tab** — check partition load state individually; releasing
   cold partitions saves memory on Cloud.
4. **User/Role tab** — create scoped database users instead of handing out the
   root token. `launchpad` does not automate this today; this is the place to
   do it.
5. **System View** — node-level stats; useful when a query-node is hot and you
   need to size-up before the next eval round.

For SSH-reachable remote hosts, forward the port instead of exposing it:

```bash
ssh -L 8000:localhost:8000 <jumpbox>
# then open http://localhost:8000 on your laptop
```

## Versioning

The Attu image tag is pinned to the Milvus minor version in
`docker-compose.yml` (`milvusdb/milvus:v2.6.x` ↔ `zilliz/attu:v2.6`). When you
bump Milvus, bump Attu in the same PR — mismatched majors/minors surface as
confusing "field not found" errors in the UI before they surface as API errors.
