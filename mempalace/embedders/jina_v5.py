"""ChromaDB ``EmbeddingFunction`` backed by ``jinaai/jina-embeddings-v5-text-nano``.

Rationale for this adapter
--------------------------
MemPalace's default ChromaDB embedder is ``all-MiniLM-L6-v2`` — English only
and capped at 256 tokens. For a Chinese-facing deployment this degrades
semantic retrieval to the point of uselessness; see
``docs/mempalace-chinese-integration.md`` for the validation report (avg
cross-lingual cosine 0.87, classification accuracy 85.7% vs. 4% with the
legacy English regexes).

Implementation notes
--------------------
* We load Jina v5 via ``transformers.AutoModel`` rather than
  ``sentence_transformers.SentenceTransformer``. Jina ships a ``custom_st.py``
  module that recent ``sentence-transformers`` releases cannot import
  (``ModuleNotFoundError: custom_st``). ``AutoModel`` sidesteps the issue and
  Jina's ``encode()`` method natively supports ``task`` / ``prompt_name``.
* The model cache is keyed by ``(model_name, os.getpid())`` so worker forks
  under gunicorn/uvicorn receive their own instance; sharing a torch model
  across fork boundaries is unsafe.
* ChromaDB reuses a single ``EmbeddingFunction`` for both ``add()`` and
  ``query()``. Jina's ``document`` prompt is a sound default for both paths;
  callers that need a dedicated query-side prompt can pass ``mode="query"``
  explicitly when constructing the function.
"""

from __future__ import annotations

import logging
import os
from typing import Any, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_JINA_V5_MODEL = "jinaai/jina-embeddings-v5-text-nano"

# ``chromadb`` exposes ``EmbeddingFunction`` as a generic protocol-ish base
# class. We subclass if available so ChromaDB's config machinery recognises
# the type; otherwise we fall back to ``object`` so this module imports even
# in environments where only the tests exercise the class directly.
try:  # pragma: no cover - exercised indirectly
    from chromadb import EmbeddingFunction as _ChromaEmbeddingFunction
except Exception:  # pragma: no cover - chromadb optional at import time
    _ChromaEmbeddingFunction = object  # type: ignore[assignment,misc]


class JinaV5EmbeddingFunction(_ChromaEmbeddingFunction):  # type: ignore[misc]
    """ChromaDB-compatible embedding function using Jina v5 retrieval task.

    Parameters
    ----------
    model_name:
        Hugging Face model id. Defaults to the 239M nano checkpoint which
        covers 100+ languages with a 32K context window.
    mode:
        ``"document"`` (default) encodes inputs with the document-side prompt,
        suitable for both indexing and most query paths. Pass ``"query"`` to
        use the query-side prompt when you know the function will only be
        invoked for search.
    """

    _cache: dict = {}

    def __init__(
        self,
        model_name: str = DEFAULT_JINA_V5_MODEL,
        mode: str = "document",
    ) -> None:
        if mode not in ("document", "query"):
            raise ValueError(f"mode must be 'document' or 'query', got {mode!r}")
        self.model_name = model_name
        self.mode = mode

    # --- ChromaDB metadata hooks -------------------------------------------------

    @staticmethod
    def name() -> str:
        """Stable identifier used by ChromaDB's EmbeddingFunction registry."""
        return "jina_v5"

    def get_config(self) -> dict:
        return {"model_name": self.model_name, "mode": self.mode}

    @classmethod
    def build_from_config(cls, config: dict) -> "JinaV5EmbeddingFunction":
        return cls(
            model_name=config.get("model_name", DEFAULT_JINA_V5_MODEL),
            mode=config.get("mode", "document"),
        )

    def default_space(self) -> str:
        """Jina v5 vectors are L2-normalised; cosine is the correct distance."""
        return "cosine"

    def supported_spaces(self) -> List[str]:
        return ["cosine", "l2", "ip"]

    @staticmethod
    def max_tokens() -> int:
        return 32768

    # --- Model loading -----------------------------------------------------------

    def _get_model(self) -> Any:
        key = (self.model_name, os.getpid())
        cached = self.__class__._cache.get(key)
        if cached is not None:
            return cached
        try:
            from transformers import AutoModel  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "transformers is required for JinaV5EmbeddingFunction; install "
                "mempalace[jina] or `pip install transformers>=4.57 torch`"
            ) from exc
        logger.info(
            "Loading Jina v5 embedding model %r (pid=%d)", self.model_name, os.getpid()
        )
        model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True)
        self.__class__._cache[key] = model
        return model

    # --- EmbeddingFunction contract ----------------------------------------------

    def __call__(self, input: List[str]) -> List[List[float]]:  # type: ignore[override]
        # ChromaDB sometimes passes an empty list on flushes; short-circuit.
        if not input:
            return []
        texts = list(input)
        model = self._get_model()

        # Jina's encode accepts ``task`` / ``prompt_name``; fall back to a
        # plain call if a future checkpoint drops the kwargs.
        try:
            vecs = model.encode(texts, task="retrieval", prompt_name=self.mode)
        except TypeError:  # pragma: no cover - defensive, keeps mocks happy
            vecs = model.encode(texts)

        if hasattr(vecs, "tolist"):
            return vecs.tolist()
        return [list(v) if not isinstance(v, list) else v for v in vecs]

    # --- Convenience encoder for internal callers --------------------------------

    def encode(
        self,
        texts: List[str],
        mode: Optional[str] = None,
    ) -> List[List[float]]:
        """Encode texts with a specific mode without mutating the instance.

        Used by :mod:`mempalace.embedders.classifier` to keep a single model
        instance in memory while switching between ``document`` anchors and
        ``query`` inputs.
        """
        if mode is None or mode == self.mode:
            return self.__call__(texts)
        if not texts:
            return []
        model = self._get_model()
        try:
            vecs = model.encode(list(texts), task="retrieval", prompt_name=mode)
        except TypeError:
            vecs = model.encode(list(texts))
        if hasattr(vecs, "tolist"):
            return vecs.tolist()
        return [list(v) if not isinstance(v, list) else v for v in vecs]


def _register_with_chromadb() -> None:
    """Register ``JinaV5EmbeddingFunction`` in ChromaDB's EF registry.

    ChromaDB 1.x exposes the registry in ``chromadb.utils.embedding_functions``
    (dict ``known_embedding_functions`` plus helper ``register_embedding_function``).
    Older layouts occasionally re-exported it via ``chromadb.api.types``; we
    try both locations so the registration works across minor versions.

    Without this registration ChromaDB warns ``Could not reconstruct embedding
    function jina_v5`` on reopen and falls back to ``None``. Any caller that
    bypasses :class:`ChromaBackend.get_collection` would then write vectors
    with ChromaDB's default ``all-MiniLM-L6-v2``, silently mixing vector
    spaces. Registering on import keeps the on-disk representation round-trip
    safe regardless of entry point.
    """
    candidate_modules = []
    try:  # pragma: no cover - exercised at import time
        from chromadb.utils import embedding_functions as _ef_mod  # type: ignore

        candidate_modules.append(_ef_mod)
    except Exception:
        pass
    try:  # pragma: no cover - version shim
        from chromadb.api import types as _api_types  # type: ignore

        candidate_modules.append(_api_types)
    except Exception:
        pass

    registered = False
    for mod in candidate_modules:
        register_fn = getattr(mod, "register_embedding_function", None)
        if callable(register_fn):
            try:
                register_fn(JinaV5EmbeddingFunction)
                registered = True
                break
            except Exception:
                pass
    if registered:
        return
    # Fallback: write the raw dict entry when the helper is unavailable or
    # refused the class (e.g. stricter validation in newer releases).
    for mod in candidate_modules:
        registry = getattr(mod, "known_embedding_functions", None)
        if isinstance(registry, dict):
            registry.setdefault("jina_v5", JinaV5EmbeddingFunction)


_register_with_chromadb()
