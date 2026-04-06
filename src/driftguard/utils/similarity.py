import numpy as np


def cosine_similarity(a, b) -> float:
    """
    Compute cosine similarity between two vectors.
    Returns 0.0 safely if either vector is zero-norm.
    """

    denom = np.linalg.norm(a) * np.linalg.norm(b)

    if denom == 0:
        return 0.0

    return float(np.dot(a, b) / denom)