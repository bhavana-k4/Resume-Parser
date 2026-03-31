"""
modules/embedder.py
===================
Generates dense vector embeddings using the Sentence Transformers library.

Model: all-MiniLM-L6-v2
  - 384-dimensional embeddings
  - ~22M parameters — runs fast on CPU (< 1s per document)
  - Downloads automatically on first use (~90 MB, cached in ~/.cache)
  - Trained on 1B+ sentence pairs for semantic similarity tasks

Why embeddings beat TF-IDF:
  TF-IDF: "Python developer" ≠ "software engineer in Python" (different words)
  Embeddings: both map to nearly identical 384-dim vectors (same meaning)
"""
import logging
import numpy as np
from functools import lru_cache

log = logging.getLogger(__name__)

MODEL_NAME = "all-MiniLM-L6-v2"
_MODEL = None   # loaded once, reused for all requests


def _load_model():
    """Load the sentence transformer model (cached after first call)."""
    global _MODEL
    if _MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
            log.info(f"Loading embedding model: {MODEL_NAME}")
            _MODEL = SentenceTransformer(MODEL_NAME)
            log.info("Embedding model loaded successfully.")
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed.\n"
                "Run: pip install sentence-transformers"
            )
        except Exception as e:
            log.error(f"Failed to load embedding model: {e}")
            raise
    return _MODEL


def embed_text(text: str) -> np.ndarray:
    """
    Embed a single text string.

    Returns:
        numpy array of shape (384,) — a normalised dense vector.

    The model uses mean pooling over all token embeddings and then
    L2-normalises the result, so cosine similarity = dot product.
    """
    model  = _load_model()
    vector = model.encode(text, normalize_embeddings=True, show_progress_bar=False)
    return np.array(vector, dtype=np.float32)


def embed_chunks(chunks: list[str]) -> np.ndarray:
    """
    Embed multiple text chunks and return a single document-level embedding
    by averaging all chunk vectors (mean pooling at document level).

    This is critical for long resumes that exceed 512 tokens:
    instead of truncating, we embed each chunk and average them.

    Returns:
        numpy array of shape (384,) — the document embedding.
    """
    if not chunks:
        return np.zeros(384, dtype=np.float32)

    model    = _load_model()
    vectors  = model.encode(chunks, normalize_embeddings=True, show_progress_bar=False)
    doc_vec  = np.mean(vectors, axis=0)

    # Re-normalise after averaging (mean of unit vectors is not unit)
    norm = np.linalg.norm(doc_vec)
    if norm > 0:
        doc_vec = doc_vec / norm

    return doc_vec.astype(np.float32)


def embed_skill_list(skills: list[str]) -> dict[str, np.ndarray]:
    """
    Embed individual skill phrases. Returns a dict mapping
    each skill string to its 384-dim embedding.

    Used by the similarity engine to find semantically similar skills
    even when exact strings don't match (e.g. "ML" ↔ "machine learning").
    """
    if not skills:
        return {}
    model   = _load_model()
    vectors = model.encode(skills, normalize_embeddings=True, show_progress_bar=False)
    return {skill: vec for skill, vec in zip(skills, vectors)}


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Cosine similarity between two L2-normalised vectors.
    Since both are unit vectors: cosine_sim = dot product.
    Returns a float in [-1, 1], typically [0, 1] for text.
    """
    return float(np.dot(v1, v2))