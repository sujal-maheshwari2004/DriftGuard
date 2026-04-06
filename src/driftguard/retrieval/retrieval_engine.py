from datetime import datetime, UTC

from driftguard.models.response import Warning, RetrievalResponse


class RetrievalEngine:
    """
    Converts graph memory into agent-usable warnings.

    Given a context string:
    1. Finds semantically similar action nodes
    2. Walks causal chains from each match
    3. Returns deduplicated warnings sorted by confidence
    """

    def __init__(self, graph_store):

        self.graph_store = graph_store

    # =====================================================
    # MAIN ENTRYPOINT
    # =====================================================

    def query(self, context: str) -> RetrievalResponse:

        candidate_nodes = self.graph_store.find_similar_nodes(
            context,
            node_type="action",
        )

        chains = []

        for node in candidate_nodes:
            chains.extend(self.graph_store.get_related_chains(node))

        warnings = self._build_warnings(chains)

        return RetrievalResponse(
            query=context,
            warnings=warnings,
            chains=chains,
            confidence=min(1.0, 0.6 + 0.1 * len(chains)),
        )

    # =====================================================
    # BUILD WARNINGS
    # =====================================================

    def _build_warnings(self, chains: list) -> list[Warning]:
        """
        Build deduplicated warnings from chains.

        Deduplication key: (trigger, risk) pair.
        When duplicates exist, the highest-confidence one is kept.
        Results are sorted by confidence descending.
        """

        seen: dict[tuple, Warning] = {}

        for chain in chains:

            if len(chain) < 2:
                continue

            trigger = chain[0]
            risk = chain[1]
            key = (trigger, risk)

            node_data = self.graph_store.get_node(trigger)
            node_freq = node_data.get("frequency", 1)

            # Factor edge frequency into confidence when available
            edge_freq = self._get_edge_frequency(trigger, risk)
            confidence = self._confidence(node_freq, edge_freq)

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

    def _confidence(self, node_freq: int, edge_freq: int) -> float:
        """
        Confidence based on combined node + edge reinforcement.

        Edge frequency reflects how often a specific causal link
        was observed — stronger signal than node frequency alone.
        """

        combined = (node_freq + edge_freq) / 2

        if combined >= 5:
            return 0.95

        if combined >= 3:
            return 0.85

        if combined >= 2:
            return 0.75

        return 0.60
