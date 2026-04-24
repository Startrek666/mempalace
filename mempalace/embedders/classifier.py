"""Embedding-based anchor classifier for language-agnostic routing.

This module replaces the English-only keyword heuristics in
``general_extractor`` (5 memory types) and in hall routing (6 halls) with a
cross-lingual anchor matcher. It reuses the Jina v5 nano model that the
ChromaDB backend already loads, so the marginal cost is a handful of extra
``encode()`` calls.

Validation (see ``docs/mempalace-chinese-integration.md``):

* 5 memory types — 88.0% accuracy on the Chinese test set (vs. 4% with the
  existing English regexes).
* 6 halls — 83.3% accuracy on the Chinese test set.
* Cross-lingual alignment avg cosine 0.87 between English anchor and Chinese
  paraphrase.

The anchor strings below are the exact ones the validation script used; do
not edit them casually — re-run the validation harness in
``lemo-inference/scripts/test_embedding_classifier.py`` when tweaking.
"""

from __future__ import annotations

import logging
import math
import threading
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

MEMORY_TYPE_DESCRIPTIONS: Dict[str, str] = {
    "decision": (
        "We decided to use this technology after weighing the trade-offs. "
        "Choosing between options, selecting a framework, tool, or architecture. "
        "We chose X over Y because of these reasons. The strategy is to use this approach. "
        "Team agreed on this design. We will go with this solution for the project."
    ),
    "preference": (
        "This is my personal coding habit and style preference that I always follow. "
        "My rule is to always do X and never do Y. My convention is to use this pattern. "
        "I personally prefer this way of working. Please always follow my preference. "
        "My standard practice, my individual style, something I habitually do or avoid."
    ),
    "milestone": (
        "Finally achieved a major breakthrough after a long effort! "
        "First time ever successfully doing this. The system is now working. "
        "Successfully shipped version and deployed to production. Big improvement milestone. "
        "Significant accomplishment reached. Major progress. Users growing."
    ),
    "problem": (
        "A bug or error was found causing the service to crash or fail. "
        "Something is broken and not working correctly. Exception thrown repeatedly. "
        "Root cause of the problem was identified. Performance dropped significantly. "
        "Error messages in logs, service unavailable, data corruption issue."
    ),
    "emotional": (
        "I feel deeply scared, worried, exhausted, sad, or emotionally overwhelmed. "
        "Personal inner feelings and emotional vulnerability. Crying, missing someone. "
        "Anxiety, stress, loneliness, or joy in personal life. Relationship emotions. "
        "This is a personal emotional experience unrelated to work outcomes."
    ),
}

HALL_DESCRIPTIONS: Dict[str, str] = {
    "emotions": (
        "Personal feelings and emotional state: fear, anxiety, happiness, sadness, anger. "
        "Feeling overwhelmed, crying, emotionally vulnerable, mood and mental wellbeing. "
        "Inner emotional experience, stress and worry in personal life."
    ),
    "technical": (
        "Writing or debugging code, programming languages, software bugs and errors. "
        "API design, database queries, server infrastructure, deployment pipeline. "
        "Technical engineering work, system architecture, performance optimization."
    ),
    "memory": (
        "Saving this information to memory for future recall and retrieval. "
        "Archive this important note, store in knowledge base, remember for next time. "
        "This conversation should be stored, memory palace entry, record for reference."
    ),
    "identity": (
        "My name is, who I am as a unique individual, my personal identity and character. "
        "My personality traits, life philosophy, core values and beliefs as a person. "
        "My personal background story, what makes me who I am, self-introduction."
    ),
    "family": (
        "My child daughter or son did something today. Family members at home. "
        "Parents, siblings, spouse, kids growing up, parenting and family life. "
        "Relatives and loved ones, family health, home and household matters."
    ),
    "creative": (
        "Creating a game, writing a song or music, making art, drawing, composing. "
        "Writing fiction or stories, film making, creative inspiration from movies. "
        "Design project, UI visual design, artistic creation, creative hobby work."
    ),
}


def _cosine(a, b) -> float:
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    denom = math.sqrt(na) * math.sqrt(nb)
    return dot / (denom + 1e-9)


class EmbeddingClassifier:
    """Thin wrapper that embeds anchor descriptions once, then classifies by cosine.

    The classifier is deliberately stateless outside its anchor cache; callers
    are expected to pin one instance per category set (memory types, halls,
    etc.) for the lifetime of the process.
    """

    def __init__(
        self,
        descriptions: Dict[str, str],
        model_name: Optional[str] = None,
        threshold: float = 0.20,
    ) -> None:
        if not descriptions:
            raise ValueError("descriptions must be a non-empty mapping")
        self.descriptions = dict(descriptions)
        self.model_name = model_name
        self.threshold = threshold
        self._anchors: Optional[Dict[str, list]] = None
        self._lock = threading.Lock()

    # --- Embedder access --------------------------------------------------------

    def _get_embedder(self):
        # Import locally so classifier module stays import-cheap when nobody
        # calls ``classify`` — helps ``mempalace`` CLI start-up.
        from .jina_v5 import DEFAULT_JINA_V5_MODEL, JinaV5EmbeddingFunction

        return JinaV5EmbeddingFunction(
            model_name=self.model_name or DEFAULT_JINA_V5_MODEL,
            mode="document",
        )

    def _ensure_anchors(self) -> None:
        if self._anchors is not None:
            return
        with self._lock:
            if self._anchors is not None:
                return
            embedder = self._get_embedder()
            labels = list(self.descriptions.keys())
            texts = [self.descriptions[label] for label in labels]
            vecs = embedder.encode(texts, mode="document")
            self._anchors = {label: vec for label, vec in zip(labels, vecs)}

    # --- Classification ---------------------------------------------------------

    def classify(self, text: str) -> Tuple[str, float]:
        """Return ``(label, score)``. Returns ``("unknown", score)`` below threshold."""
        if not text or not text.strip():
            return "unknown", 0.0
        self._ensure_anchors()
        assert self._anchors is not None  # for type-checkers
        embedder = self._get_embedder()
        vec = embedder.encode([text], mode="query")[0]
        sims = {label: _cosine(vec, anchor) for label, anchor in self._anchors.items()}
        best = max(sims, key=sims.get)
        score = sims[best]
        if score < self.threshold:
            return "unknown", score
        return best, score


# --- Process-wide singletons -----------------------------------------------------

_memory_classifier: Optional[EmbeddingClassifier] = None
_hall_classifier: Optional[EmbeddingClassifier] = None
_singleton_lock = threading.Lock()


def get_memory_type_classifier(
    model_name: Optional[str] = None,
) -> EmbeddingClassifier:
    """Return the singleton classifier for the 5 ``general_extractor`` memory types."""
    global _memory_classifier
    if _memory_classifier is not None:
        return _memory_classifier
    with _singleton_lock:
        if _memory_classifier is None:
            _memory_classifier = EmbeddingClassifier(
                MEMORY_TYPE_DESCRIPTIONS, model_name=model_name
            )
    return _memory_classifier


def get_hall_classifier(
    model_name: Optional[str] = None,
) -> EmbeddingClassifier:
    """Return the singleton classifier for the 6-hall content router."""
    global _hall_classifier
    if _hall_classifier is not None:
        return _hall_classifier
    with _singleton_lock:
        if _hall_classifier is None:
            _hall_classifier = EmbeddingClassifier(
                HALL_DESCRIPTIONS, model_name=model_name
            )
    return _hall_classifier
