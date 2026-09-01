import numpy as np

from driftguard.config import DEFAULT_SETTINGS, DriftGuardSettings
from driftguard.embedding.embedding_engine import EmbeddingEngine
from driftguard.logging_config import get_logger
from driftguard.utils.normalization import normalize_text
from driftguard.utils.similarity import cosine_similarity


logger = get_logger(__name__)


def literal_tokens(text: str) -> frozenset[str]:
    """
    Return the tokens that carry an identity rather than a meaning.

    Embeddings score "delete user 1" and "delete user 2" at ~0.93 because the
    two differ only in a digit, which the encoder barely notices. Merging them
    silently destroys one of the memories, so any token containing a digit
    (ids, counts, versions, ports) is treated as a literal that must match
    exactly before two nodes can be considered the same.
    """

    return frozenset(
        token
        for token in text.split()
        if any(character.isdigit() for character in token)
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

    def __init__(
        self,
        *,
        settings: DriftGuardSettings | None = None,
        embedding_engine: EmbeddingEngine | None = None,
    ):
        self.settings = settings or DEFAULT_SETTINGS
        self.embedding_engine = embedding_engine or EmbeddingEngine(
            model_name=self.settings.embedding_model_name,
            device=self.settings.embedding_device,
        )
        logger.info(
            "Merge engine ready with embedding_model=%s",
            self.embedding_engine.model_name(),
        )

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

        Candidates whose literal tokens differ from the query's are rejected
        before scoring — see literal_tokens().
        """

        query_literals = literal_tokens(text)

        candidates = [
            node
            for node in graph.nodes
            if graph.nodes[node]["type"] == node_type
            and literal_tokens(node) == query_literals
        ]

        if not candidates:
            logger.debug("No candidates available for node_type=%r", node_type)
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

        logger.debug(
            "Best match lookup text=%r node_type=%r candidates=%d matched=%r score=%.4f",
            text,
            node_type,
            len(candidates),
            best_node,
            best_score,
        )
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
        min_similarity: float = 0.0,
        include_scores: bool = False,
    ) -> list[str] | list[tuple[str, float]]:
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
            logger.debug("Top-k lookup found no candidates for node_type=%r", node_type)
            return []

        query_emb = self.embed(text)

        # Stack all embeddings into a matrix for vectorised similarity
        embeddings = np.stack(
            [graph.nodes[n]["embedding"] for n in candidates]
        )

        scores = embeddings @ query_emb  # cosine sim (embeddings are normalized)

        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []

        for index in top_indices:
            score = float(scores[index])

            if score < min_similarity:
                continue

            if include_scores:
                results.append((candidates[index], score))
            else:
                results.append(candidates[index])

        logger.debug(
            "Top-k lookup text=%r node_type=%r candidates=%d returned=%d min_similarity=%.2f include_scores=%s",
            text,
            node_type,
            len(candidates),
            len(results),
            min_similarity,
            include_scores,
        )
        return results

    # =====================================================
    # INTERNAL: THRESHOLD LOOKUP
    # =====================================================

    def _get_threshold(self, node_type: str) -> float:
        return self.settings.threshold_for(node_type)
