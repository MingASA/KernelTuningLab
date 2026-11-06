#!/usr/bin/env bash
# A destination must be supplied explicitly; this script never uploads anywhere.
set -euo pipefail
: "${KT_BACKUP_DIR:?Set KT_BACKUP_DIR to an authorized mounted/local backup directory}"
KT_REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
mkdir -p -- "$KT_BACKUP_DIR"
for KT_ARCHIVE in "$KT_REPO_ROOT"/artifacts/*evidence*.tar.gz; do
  [[ -f "$KT_ARCHIVE" && -f "$KT_ARCHIVE.sha256" ]] || continue
  # Phase2's first seal is retained locally for provenance; the verified v2 is
  # the delivery archive. Do not accidentally select the superseded version.
  if [[ "$(basename -- "$KT_ARCHIVE")" == research-phase2-evidence-20261026.tar.gz \
        && -f "$KT_REPO_ROOT/artifacts/research-phase2-evidence-20261026-v2.tar.gz" \
        && -f "$KT_REPO_ROOT/artifacts/research-phase2-evidence-20261026-v2.tar.gz.sha256" ]]; then
    continue
  fi
  (cd -- "$(dirname -- "$KT_ARCHIVE")" && sha256sum -c -- "$(basename -- "$KT_ARCHIVE.sha256")")
  cp --no-clobber -- "$KT_ARCHIVE" "$KT_ARCHIVE.sha256" "$KT_BACKUP_DIR/"
  (cd -- "$KT_BACKUP_DIR" && sha256sum -c -- "$(basename -- "$KT_ARCHIVE.sha256")")
done
