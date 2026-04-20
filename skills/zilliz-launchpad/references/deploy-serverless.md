# Deploying to Zilliz Cloud Serverless

**Status**: reference only. Phase 6 Deploy will automate cluster creation in a later change. Today you *connect* to Zilliz Cloud from Phase 4 — this doc explains how to pick the cluster.

## Connecting — preferred path (with `zilliz` CLI)

Install the [zilliz CLI](https://github.com/zilliztech/zilliz-cli) (≥ 0.3.0) and log in once:

```bash
zilliz auth login
zilliz cluster list
```

Then in Phase 2 set `--deployment zilliz-serverless`. Configure auto-discovers your Serverless clusters via `zilliz cluster list`, writes the chosen `cluster_id` and `target_uri` into `configure.json`, and Phase 4 uses `zilliz cluster describe` as a pre-flight (fails fast if the cluster is paused or still provisioning).

Token resolution: if `ZILLIZ_TOKEN` is unset but the CLI is authenticated, the launchpad reads the session token from `zilliz auth whoami`.

## Connecting — manual path (no CLI)

1. Create a Serverless cluster at <https://cloud.zilliz.com>.
2. Copy the cluster's **Public Endpoint** and an **API key**.
3. Export:
   ```bash
   export ZILLIZ_TOKEN=<api-key>
   ```
4. In Phase 2 set `--deployment zilliz-serverless`; Phase 3's `plan.json` will contain a `target_uri` with a placeholder host — replace it with your endpoint URL before running Phase 4.

No CLI means no pre-flight and no bulk-import routing. Everything else works.

## When to pick Serverless

- Prototypes and small-to-medium workloads
- Elastic traffic; pay-per-use
- No need to manage capacity

## Constraints

- No GPU index types
- `DISKANN` available (often the default for large data on Serverless)
- Some admin-heavy operations (compaction, flush tuning) are managed by the service

Cluster provisioning is still manual — Phase 6 Deploy will automate this via `zilliz cluster create` in a later change.
