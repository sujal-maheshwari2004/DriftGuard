import numpy as np

from DriftGuard.embedding.embedding_engine import EmbeddingEngine
from DriftGuard.utils.normalization import normalize_text
from DriftGuard.utils.similarity import cosine_similarity

from DriftGuard.config import (
    SIM_THRESHOLD_ACTION,
    SIM_THRESHOLD_FEEDBACK,
    SIM_THRESHOLD_OUTCOME,
)


class MergeEngine:
    """
    Handles semantic node deduplication.

    Responsibilities:
    - Normalize text
    - Embed text
    - Detect semantic duplicates via cosine similarity
    - Return canonical node match or None
    """

    def __init__(self):
        self.embedding_engine = EmbeddingEngine()

    # =====================================================
    # NORMALIZATION
    # =====================================================

    def normalize(self, text: str) -> str:
        return normalize_text(text)

    # =====================================================
    # EMBEDDING
    # =====================================================

    def embed(self, text: str):
        return self.embedding_engine.embed(text)

    # =====================================================
    # NODE MATCHING (single query vs. graph)
    # =====================================================

    def find_similar_node(
        self,
        text: str,
        node_type: str,
        graph,
    ) -> str | None:
        """
        Return the best matching node if similarity exceeds threshold.
        Returns None if graph is empty or no match found.
        """

        candidates = [
            node
            for node in graph.nodes
            if graph.nodes[node]["type"] == node_type
        ]

        if not candidates:
            return None

        query_emb = self.embed(text)
        threshold = self._get_threshold(node_type)

        best_node = None
        best_score = threshold  # must beat threshold to qualify

        for node in candidates:
            score = cosine_similarity(
                query_emb,
                graph.nodes[node]["embedding"],
            )

            if score > best_score:
                best_score = score
                best_node = node

        return best_node

    # =====================================================
    # TOP-K LOOKUP (batch, used by retrieval engine)
    # =====================================================

    def find_top_k_similar(
        self,
        text: str,
        graph,
        node_type: str = None,
        top_k: int = 5,
    ) -> list[str]:
        """
        Return top-k most similar nodes.

        Uses matrix operations for efficiency when graph is large.
        """

        candidates = [
            node
            for node in graph.nodes
            if node_type is None or graph.nodes[node]["type"] == node_type
        ]

        if not candidates:
            return []

        query_emb = self.embed(text)

        # Stack all embeddings into a matrix for vectorised similarity
        embeddings = np.stack(
            [graph.nodes[n]["embedding"] for n in candidates]
        )

        scores = embeddings @ query_emb  # cosine sim (embeddings are normalized)

        top_indices = np.argsort(scores)[::-1][:top_k]

        return [candidates[i] for i in top_indices]

    # =====================================================
    # INTERNAL: THRESHOLD LOOKUP
    # =====================================================

    def _get_threshold(self, node_type: str) -> float:

        return {
            "action": SIM_THRESHOLD_ACTION,
            "feedback": SIM_THRESHOLD_FEEDBACK,
            "outcome": SIM_THRESHOLD_OUTCOME,
        }.get(node_type, 0.85)