#!/usr/bin/env bash
set -euo pipefail

python /app/scripts/ensure_models.py

exec "$@"
