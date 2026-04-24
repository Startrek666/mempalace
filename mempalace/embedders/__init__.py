"""Embedding model adapters and palace-level model configuration for MemPalace.

This package hosts:

* :class:`JinaV5EmbeddingFunction` — a ChromaDB ``EmbeddingFunction`` wrapping
  ``jinaai/jina-embeddings-v5-text-nano`` with fork-safe caching.
* Palace sidecar helpers (``.palace_meta.json``) for recording which embedding
  model was used to mine a palace, so mismatches between write-time and
  read-time embedders can be detected and surfaced as a hard error rather than
  silently corrupting semantic retrieval.
* :class:`EmbeddingClassifier` — a language-agnostic anchor-based classifier
  that reuses the already-loaded Jina model to replace English-only keyword
  heuristics in ``general_extractor`` and hall routing.
"""

from .jina_v5 import DEFAULT_JINA_V5_MODEL, JinaV5EmbeddingFunction
from .palace_meta import (
    PALACE_META_FILENAME,
    read_palace_embedding_model,
    write_palace_embedding_model,
)

__all__ = [
    "DEFAULT_JINA_V5_MODEL",
    "JinaV5EmbeddingFunction",
    "PALACE_META_FILENAME",
    "read_palace_embedding_model",
    "write_palace_embedding_model",
]
