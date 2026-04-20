# Deploying BYOC (Bring Your Own Cloud)

**Status**: reference only; Deploy phase is future work.

Zilliz BYOC provisions a managed Zilliz control plane inside **your** AWS / GCP / Azure account. Data stays in your VPC.

## When to pick BYOC

- Strict data-residency / sovereignty requirements
- Existing committed-spend with a hyperscaler you want to draw down
- Compliance regimes that forbid multi-tenant cloud storage

## What the launchpad covers today

- The Phase 4 `execute` will connect to a BYOC endpoint once you supply its URI and token. Same shape as Serverless / Dedicated.

What it does **not** do:

- Does not set up the BYOC control plane
- Does not provision IAM roles, VPC endpoints, or KMS keys

Those are one-time operations documented at <https://docs.zilliz.com/docs/byoc>.
