"""Shared identities for comparable observed measurement protocols."""

from __future__ import annotations

from .models import MeasurementRecord, stable_id


def source_context(record: MeasurementRecord) -> dict:
    """Seeds and experiment IDs may differ; timing protocols may not.

    An upstream file identity distinguishes imported experiments even when
    their available environment metadata is sparse.
    """
    manifest = record.metadata.get("source_manifest", {})
    source_identity = None
    if record.provenance == "imported":
        source_identity = (record.metadata.get("source_file_sha256")
                           or (manifest.get("git_blob_sha") if isinstance(manifest, dict) else None)
                           or record.metadata.get("source_url"))
    return {"provenance": record.provenance, "measurement_mode": record.measurement_mode,
            "protocol": record.metadata.get("protocol", {}), "environment": record.environment,
            "warmup": record.warmup, "repetitions": record.repetitions, "source_identity": source_identity}


def source_context_id(record: MeasurementRecord) -> str:
    """Identity of compatible provenance, environment, and timing protocol."""
    return stable_id(source_context(record))


__all__ = ["source_context", "source_context_id"]
