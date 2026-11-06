#!/usr/bin/env bash
# Thin launcher for an already installed private Web environment.
set -euo pipefail
: "${KTL_WEB_ENV:?Set KTL_WEB_ENV to the absolute service venv directory}"
KTL_WORKSPACE="${KTL_WORKSPACE:-$HOME/.local/share/ktlab/workbench}"
KTL_WEB_PORT="${KTL_WEB_PORT:-8765}"
args=(serve --workspace "$KTL_WORKSPACE" --port "$KTL_WEB_PORT")
if [[ -n "${KTL_SERVER_CONFIG:-}" ]]; then
  args+=(--config "$KTL_SERVER_CONFIG")
fi
exec "$KTL_WEB_ENV/bin/ktlab" "${args[@]}" "$@"
