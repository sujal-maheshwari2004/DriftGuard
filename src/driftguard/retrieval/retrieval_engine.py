from datetime import UTC, datetime

from driftguard.logging_config import get_logger
from driftguard.models.response import Reinforcement, RetrievalResponse, Warning


logger = get_logger(__name__)


class RetrievalEngine:
    """
    Converts graph memory into agent-usable warnings and reinforcements.

    Given a context string:
    1. Finds semantically similar action nodes
    2. Walks causal chains from each match
    3. Returns deduplicated warnings (from the mistake graph) and
       reinforcements (from the optional success graph), sorted by confidence
    """

    def __init__(
        self,
        graph_store,
        success_graph_store=None,
        *,
        top_k: int = 5,
        min_similarity: float = 0.0,
        recency_weight: float = 0.15,
        metrics=None,
    ):

        self.graph_store = graph_store
        self.success_graph_store = success_graph_store
        self.top_k = top_k
        self.min_similarity = min_similarity
        self.recency_weight = recency_weight
        self.metrics = metrics
        logger.info("Retrieval engine initialized")

    # =====================================================
    # MAIN ENTRYPOINT
    # =====================================================

    def query(self, context: str) -> RetrievalResponse:

        chain_matches = self._collect_chain_matches(self.graph_store, context)
        chains = [chain for chain, _ in chain_matches]
        warnings = self._build_signals(chain_matches, self.graph_store, Warning, "risk")
        confidence = max((warning.confidence for warning in warnings), default=0.0)

        reinforcements = []
        if self.success_graph_store is not None:
            success_chain_matches = self._collect_chain_matches(
                self.success_graph_store, context
            )
            reinforcements = self._build_signals(
                success_chain_matches,
                self.success_graph_store,
                Reinforcement,
                "recommendation",
            )

        logger.info(
            "Retrieval query context=%r chains=%d warnings=%d reinforcements=%d confidence=%.2f",
            context,
            len(chains),
            len(warnings),
            len(reinforcements),
            confidence,
        )
        if self.metrics is not None:
            self.metrics.record_review(
                warnings_count=len(warnings),
                confidence=confidence,
            )

        return RetrievalResponse(
            query=context,
            warnings=warnings,
            chains=chains,
            confidence=confidence,
            reinforcements=reinforcements,
        )

    # =====================================================
    # COLLECT CHAIN MATCHES
    # =====================================================

    def _collect_chain_matches(
        self,
        graph_store,
        context: str,
    ) -> list[tuple[list[str], float]]:
        candidate_matches = graph_store.find_similar_nodes(
            context,
            node_type="action",
            top_k=self.top_k,
            min_similarity=self.min_similarity,
            include_scores=True,
        )

        chain_matches = []

        for node, similarity_score in candidate_matches:
            for chain in graph_store.get_related_chains(node):
                chain_matches.append((chain, similarity_score))

        return chain_matches

    # =====================================================
    # BUILD SIGNALS (WARNINGS OR REINFORCEMENTS)
    # =====================================================

    def _build_signals(
        self,
        chain_matches: list[tuple[list[str], float]],
        graph_store,
        signal_cls,
        second_field_name: str,
    ):
        """
        Build deduplicated signals from chains.

        Deduplication key: (trigger, second-node) pair.
        When duplicates exist, the highest-confidence one is kept.
        Results are sorted by confidence descending.
        """

        seen: dict[tuple, object] = {}

        for chain, similarity_score in chain_matches:

            if len(chain) < 2:
                continue

            trigger = chain[0]
            second = chain[1]
            key = (trigger, second)

            node_data = graph_store.get_node(trigger)
            node_freq = node_data.get("frequency", 1)
            recency_score = self._recency_score(node_data.get("last_seen"))

            # Factor edge frequency into confidence when available
            edge_freq = self._get_edge_frequency(graph_store, trigger, second)
            confidence = self._confidence(
                node_freq,
                edge_freq,
                similarity_score,
                recency_score=recency_score,
            )

            if key not in seen or confidence > seen[key].confidence:
                seen[key] = signal_cls(
                    trigger=trigger,
                    **{second_field_name: second},
                    frequency=node_freq,
                    confidence=confidence,
                )

        return sorted(
            seen.values(),
            key=lambda signal: signal.confidence,
            reverse=True,
        )

    # =====================================================
    # EDGE FREQUENCY LOOKUP
    # =====================================================

    def _get_edge_frequency(self, graph_store, src: str, dst: str) -> int:
        """
        Return edge frequency between two nodes, or 1 if edge absent.
        """

        graph = graph_store.graph

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
        *,
        recency_score: float | None = None,
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

        baseline = 0.65 * reinforcement_confidence + 0.35 * similarity_score

        if recency_score is None:
            return min(1.0, baseline)

        recency_weight = max(0.0, min(self.recency_weight, 0.5))
        remaining_weight = 1.0 - recency_weight
        reinforcement_weight = remaining_weight * 0.65
        similarity_weight = remaining_weight * 0.35

        return min(
            1.0,
            reinforcement_weight * reinforcement_confidence
            + similarity_weight * similarity_score
            + recency_weight * recency_score,
        )

    def _recency_score(self, last_seen) -> float | None:
        if not isinstance(last_seen, datetime):
            return None

        age = datetime.now(UTC) - last_seen

        if age.days <= 1:
            return 1.0
        if age.days <= 7:
            return 0.85
        if age.days <= 30:
            return 0.70
        return 0.50
