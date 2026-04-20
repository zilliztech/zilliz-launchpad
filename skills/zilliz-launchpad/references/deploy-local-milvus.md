# Deploying locally: Milvus Standalone

The launchpad runs Milvus Standalone via Docker Compose, following the official guide: <https://milvus.io/docs/install_standalone-docker.md>.

Milvus Lite is **not** supported in this MVP — use Standalone locally and Zilliz Cloud for production.

## Start / stop

```bash
cd skills/zilliz-launchpad/scripts
./start_milvus.sh up        # boot etcd + MinIO + Milvus
./start_milvus.sh status    # docker compose ps
./start_milvus.sh health    # curl /healthz
./start_milvus.sh logs      # tail logs (-f)
./start_milvus.sh down      # stop, keep volumes
./start_milvus.sh clean     # stop AND wipe ./volumes/ (destroys all collections)
```

Expose after `up`:

| Endpoint      | URL                          | Purpose              |
| ---           | ---                          | ---                  |
| Milvus SDK    | `http://localhost:19530`     | pymilvus / launchpad |
| Milvus health | `http://localhost:9091/healthz` | readiness probe    |
| MinIO console | `http://localhost:9001`      | `minioadmin` / `minioadmin` |
| MinIO S3 API  | `http://localhost:9000`      | object storage       |

## Requirements

- Docker Desktop (or compatible engine) running locally
- `docker compose` v2 (bundled with Docker Desktop ≥ 4.x)
- ~4 GB free disk space for first-run images and volumes
- Ports `19530`, `9091`, `9000`, `9001` free

Data persists under `./volumes/` (override with `DOCKER_VOLUME_DIRECTORY`).

## Version pinning

`docker-compose.yml` pins the upstream stable release:

- `milvusdb/milvus:v2.6.15`
- `quay.io/coreos/etcd:v3.5.25`
- `minio/minio:RELEASE.2024-05-28T17-19-04Z`

To upgrade Milvus, bump the `standalone.image` tag and re-run `up`. Check the [Milvus release notes](https://github.com/milvus-io/milvus/releases) before upgrading major versions — data-layout changes can require migration.

## Common issues

- **Port 19530 / 9001 in use** — usually a stale Milvus container from a previous run. Try `docker ps | grep milvus` and stop it, or `./start_milvus.sh clean`.
- **First boot is slow** — Milvus reports healthy after ~30-60s; the compose file's `start_period: 90s` accounts for this.
- **Image pull slow** — first run downloads ~1.5 GB across etcd, MinIO, and Milvus.
- **`docker compose` not found** — you likely have the legacy `docker-compose` (v1). Install the v2 plugin.
- **Reset all data** — `./start_milvus.sh clean` stops everything and removes `./volumes/`.

## Alternatives (documented but not automated)

The official guide also ships a convenience installer:

```bash
curl -sfL https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh -o standalone_embed.sh
bash standalone_embed.sh start
```

This vendors a single-container flavor of Milvus. The launchpad prefers the full compose stack because it matches the Zilliz Cloud data-flow more closely (separate metadata and object stores).
