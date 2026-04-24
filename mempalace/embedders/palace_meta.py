"""Palace-level sidecar metadata for MemPalace.

The sidecar file ``{palace_path}/.palace_meta.json`` records the embedding
model the palace was mined with, plus creation/update timestamps. It is a
parallel, human-readable record of what ChromaDB also stores in the
collection metadata; the two are kept in sync by
:class:`mempalace.backends.chroma.ChromaBackend` during ``get_collection``.

Why a sidecar on top of collection metadata
-------------------------------------------
* Tooling that migrates a palace (e.g. ``mempalace re-mine --model``) should
  be able to discover the current model without opening ChromaDB — collection
  metadata only becomes available after a successful ``PersistentClient``
  connection, which is the expensive step we may be avoiding on purpose.
* Humans inspecting a palace directory get an obvious, diff-friendly record.
* If ChromaDB drops its metadata during a reset/rebuild, we still have an
  authoritative record of what the palace was *supposed* to use.
"""

from __future__ import annotations

import datetime as _dt
import json
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

PALACE_META_FILENAME = ".palace_meta.json"


def _meta_path(palace_path: str) -> str:
    return os.path.join(palace_path, PALACE_META_FILENAME)


def read_palace_meta(palace_path: str) -> dict:
    """Return the sidecar payload as a dict, or ``{}`` if missing/unreadable."""
    path = _meta_path(palace_path)
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to read palace meta at %s: %s", path, exc)
        return {}


def read_palace_embedding_model(palace_path: str) -> Optional[str]:
    """Return the recorded embedding model name, or ``None`` if unset."""
    model = read_palace_meta(palace_path).get("embedding_model")
    return model if isinstance(model, str) and model else None


def write_palace_embedding_model(palace_path: str, model_name: str) -> None:
    """Persist ``model_name`` to the palace sidecar, preserving other fields."""
    if not model_name:
        raise ValueError("model_name must be a non-empty string")
    os.makedirs(palace_path, exist_ok=True)
    path = _meta_path(palace_path)
    data = read_palace_meta(palace_path)
    now = _dt.datetime.now(_dt.timezone.utc).isoformat()
    data["embedding_model"] = model_name
    data.setdefault("created_at", now)
    data["updated_at"] = now
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, sort_keys=True)
    except OSError as exc:
        logger.warning("Failed to write palace meta at %s: %s", path, exc)
