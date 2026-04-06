from driftguard.logging_config import get_logger
from driftguard.models.response import Warning, RetrievalResponse


logger = get_logger(__name__)


class RetrievalEngine:
    """
    Converts graph memory into agent-usable warnings.

    Given a context string:
    1. Finds semantically similar action nodes
    2. Walks causal chains from each match
    3. Returns deduplicated warnings sorted by confidence
    """

    def __init__(
        self,
        graph_store,
        *,
        top_k: int = 5,
        min_similarity: float = 0.0,
    ):

        self.graph_store = graph_store
        self.top_k = top_k
        self.min_similarity = min_similarity
        logger.info("Retrieval engine initialized")

    # =====================================================
    # MAIN ENTRYPOINT
    # =====================================================

    def query(self, context: str) -> RetrievalResponse:

        candidate_matches = self.graph_store.find_similar_nodes(
            context,
            node_type="action",
            top_k=self.top_k,
            min_similarity=self.min_similarity,
            include_scores=True,
        )

        chain_matches = []

        for node, similarity_score in candidate_matches:
            for chain in self.graph_store.get_related_chains(node):
                chain_matches.append((chain, similarity_score))

        chains = [chain for chain, _ in chain_matches]
        warnings = self._build_warnings(chain_matches)
        confidence = max((warning.confidence for warning in warnings), default=0.0)
        logger.info(
            "Retrieval query context=%r candidates=%d chains=%d warnings=%d confidence=%.2f",
            context,
            len(candidate_matches),
            len(chains),
            len(warnings),
            confidence,
        )

        return RetrievalResponse(
            query=context,
            warnings=warnings,
            chains=chains,
            confidence=confidence,
        )

    # =====================================================
    # BUILD WARNINGS
    # =====================================================

    def _build_warnings(
        self,
        chain_matches: list[tuple[list[str], float]],
    ) -> list[Warning]:
        """
        Build deduplicated warnings from chains.

        Deduplication key: (trigger, risk) pair.
        When duplicates exist, the highest-confidence one is kept.
        Results are sorted by confidence descending.
        """

        seen: dict[tuple, Warning] = {}

        for chain, similarity_score in chain_matches:

            if len(chain) < 2:
                continue

            trigger = chain[0]
            risk = chain[1]
            key = (trigger, risk)

            node_data = self.graph_store.get_node(trigger)
            node_freq = node_data.get("frequency", 1)

            # Factor edge frequency into confidence when available
            edge_freq = self._get_edge_frequency(trigger, risk)
            confidence = self._confidence(
                node_freq,
                edge_freq,
                similarity_score,
            )

            if key not in seen or confidence > seen[key].confidence:
                seen[key] = Warning(
                    trigger=trigger,
                    risk=risk,
                    frequency=node_freq,
                    confidence=confidence,
                )

        return sorted(
            seen.values(),
            key=lambda w: w.confidence,
            reverse=True,
        )

    # =====================================================
    # EDGE FREQUENCY LOOKUP
    # =====================================================

    def _get_edge_frequency(self, src: str, dst: str) -> int:
        """
        Return edge frequency between two nodes, or 1 if edge absent.
        """

        graph = self.graph_store.graph

        if graph.has_edge(src, dst):
            return graph[src][dst].get("frequency", 1)

        return 1

    # =====================================================
    # CONFIDENCE SCORING
    # =====================================================

    def _confidence(
        self,
        node_freq: int,
        edge_freq: int,
        similarity_score: float,
    ) -> float:
        """
        Confidence based on combined node + edge reinforcement.

        Edge frequency reflects how often a specific causal link
        was observed — stronger signal than node frequency alone.
        """

        combined = (node_freq + edge_freq) / 2

        if combined >= 5:
            reinforcement_confidence = 0.95
        elif combined >= 3:
            reinforcement_confidence = 0.85
        elif combined >= 2:
            reinforcement_confidence = 0.75
        else:
            reinforcement_confidence = 0.60

        return min(
            1.0,
            0.65 * reinforcement_confidence + 0.35 * similarity_score,
        )
