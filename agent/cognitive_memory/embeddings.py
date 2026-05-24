"""
Embedding model integration — local sentence-transformers with lazy loading.

Model: all-MiniLM-L6-v2 (384-dimensional vectors)
- Python-native via sentence-transformers (no Node.js dependency)
- Lazy loading: model loaded on first use, not at import time
- Batched generation for efficiency
- Graceful fallback when the package is not installed

Design:
  - is_available() checks if sentence-transformers can be imported
  - model_loaded() checks if the model is in memory
  - generate_embedding() and generate_embeddings() handle warm-up and errors
  - All vectors are Python lists of floats for JSON-serializability
"""

from typing import Optional

EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 produces 384-dim vectors

# Lazy global — populated on first use
_model: Optional[object] = None
_model_name: str = "all-MiniLM-L6-v2"


def is_available() -> bool:
    """Check if sentence-transformers can be imported.

    Returns True if the package is installed, False otherwise.
    Does NOT load the model — just checks importability.
    """
    global _model
    if _model is not None:
        return True
    try:
        import sentence_transformers  # noqa: F401
        return True
    except ImportError:
        return False


def model_loaded() -> bool:
    """Check if the embedding model has been loaded into memory.

    Returns True if generate_embedding() has been called at least once
    and the model was successfully loaded.
    """
    global _model
    return _model is not None


def _get_model():
    """Lazily load the sentence-transformers model on first use.

    Returns the model instance. Raises RuntimeError if the package
    is not installed.
    """
    global _model
    if _model is not None:
        return _model

    try:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(_model_name)
        return _model
    except ImportError:
        raise RuntimeError(
            "sentence-transformers is not installed. "
            "Install with: pip install sentence-transformers"
        )


def generate_embedding(text: str) -> list[float]:
    """Generate a single embedding vector for a text string.

    Args:
        text: The text to embed. Prepended with the standard
              sentence-transformers instruction prefix for retrieval.

    Returns:
        A list of EMBEDDING_DIM floats representing the text's embedding.

    Raises:
        RuntimeError: If sentence-transformers is not installed.
    """
    model = _get_model()
    # Encode as retrieval passage for better search quality
    vector = model.encode(text, show_progress_bar=False, convert_to_numpy=True)
    return vector.tolist()


def generate_embeddings(texts: list[str]) -> list[list[float]]:
    """Generate embeddings for a batch of texts.

    Args:
        texts: List of text strings to embed.

    Returns:
        List of embedding vectors, each a list of EMBEDDING_DIM floats.
        Returns in the same order as input texts.

    Raises:
        RuntimeError: If sentence-transformers is not installed.
    """
    if not texts:
        return []

    model = _get_model()
    vectors = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return [v.tolist() for v in vectors]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors.

    Args:
        a: First vector (list of floats).
        b: Second vector (list of floats).

    Returns:
        Cosine similarity in range [-1.0, 1.0].

    Raises:
        ValueError: If vectors have different lengths.
    """
    if len(a) != len(b):
        raise ValueError(
            f"Vectors must have the same length, got {len(a)} and {len(b)}"
        )

    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return dot / (norm_a * norm_b)
