# Deploying to Zilliz Cloud Serverless

**Status**: reference only. Deploy (Phase 6) is out of scope in the MVP — you can *connect* to Zilliz Cloud from Phase 4, but the launchpad does not provision a cluster for you.

## Connecting

1. Create a Serverless cluster at <https://cloud.zilliz.com>.
2. Copy the cluster's **Public Endpoint** and an **API key**.
3. Export:
   ```bash
   export ZILLIZ_TOKEN=<api-key>
   ```
4. In Phase 2 set `--deployment zilliz-serverless`; Phase 3's `plan.json` will contain a `target_uri` with a placeholder host — replace it with your endpoint URL before running Phase 4.

## When to pick Serverless

- Prototypes and small-to-medium workloads
- Elastic traffic; pay-per-use
- No need to manage capacity

## Constraints

- No GPU index types
- `DISKANN` available (often the default for large data on Serverless)
- Some admin-heavy operations (compaction, flush tuning) are managed by the service

TODO (Phase 6): automate cluster creation via the Zilliz Cloud API.
