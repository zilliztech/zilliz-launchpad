#!/usr/bin/env bash
# Bring up Milvus Standalone via docker-compose.
# Mirrors the official guide: https://milvus.io/docs/install_standalone-docker.md
#
# Usage:
#   ./start_milvus.sh [up|down|status|logs|health|clean]

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

cmd="${1:-up}"

_require_docker() {
  if ! command -v docker >/dev/null 2>&1; then
    cat <<EOF >&2
ERROR: Docker is not installed or not on PATH.
Install Docker Desktop (https://docs.docker.com/get-docker/) and make sure it is running.
EOF
    exit 1
  fi
  if ! docker info >/dev/null 2>&1; then
    cat <<EOF >&2
ERROR: Docker daemon is not running.
Start Docker Desktop (or your Docker service) and retry.
EOF
    exit 1
  fi
  if ! docker compose version >/dev/null 2>&1; then
    cat <<EOF >&2
ERROR: 'docker compose' (v2) is required but not available.
Docker Desktop ≥ 4.x bundles it; on Linux install the 'docker-compose-plugin' package.
EOF
    exit 1
  fi
}

_check_port() {
  local port="$1"
  if command -v lsof >/dev/null 2>&1; then
    if lsof -iTCP:"$port" -sTCP:LISTEN -n -P >/dev/null 2>&1; then
      cat <<EOF >&2
ERROR: Port $port is already in use.
Stop whatever is listening (possibly a stale Milvus container), or free the port:
  docker ps | grep milvus
EOF
      exit 1
    fi
  fi
}

case "$cmd" in
  up)
    _require_docker
    _check_port 19530
    _check_port 9091
    docker compose up -d
    cat <<EOF

Milvus Standalone is starting.
  SDK endpoint:  http://localhost:19530
  Healthz:       http://localhost:9091/healthz
  MinIO console: http://localhost:9001  (user: minioadmin  pass: minioadmin)

Wait ~30-60s for first-run health to go green, then run the launchpad CLI.
Tail logs: $0 logs
EOF
    ;;
  down)
    _require_docker
    docker compose down
    ;;
  status)
    docker compose ps
    ;;
  logs)
    docker compose logs -f --tail=200
    ;;
  health)
    docker compose ps
    echo "---"
    curl -fsS http://localhost:9091/healthz && echo "  [milvus healthz OK]" || echo "  [milvus healthz not ready]"
    ;;
  clean)
    _require_docker
    docker compose down -v
    rm -rf "${HERE}/volumes"
    echo "Removed containers, compose volumes, and ./volumes/ on disk."
    ;;
  *)
    echo "Usage: $0 [up|down|status|logs|health|clean]" >&2
    exit 2
    ;;
esac
