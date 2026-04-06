from datetime import datetime, UTC

from DriftGuard.models.response import (
    Warning,
    RetrievalResponse,
)


class RetrievalEngine:
    """
    Converts graph memory into agent-usable warnings.
    """

    def __init__(self, graph_store):

        self.graph_store = graph_store

    # =====================================================
    # MAIN ENTRYPOINT
    # =====================================================

    def query(self, context):

        candidate_nodes = self.graph_store.find_similar_nodes(
            context,
            node_type="action",
        )

        chains = []

        for node in candidate_nodes:

            chains.extend(
                self.graph_store.get_related_chains(node)
            )

        warnings = []

        for chain in chains:

            if len(chain) < 2:
                continue

            node_data = self.graph_store.get_node(
                chain[0]
            )

            warnings.append(

                Warning(

                    trigger=chain[0],

                    risk=chain[1],

                    frequency=node_data["frequency"],

                    confidence=self._confidence(
                        node_data["frequency"]
                    ),
                )
            )

        return RetrievalResponse(

            query=context,

            warnings=warnings,

            chains=chains,

            confidence=min(
                1.0,
                0.6 + 0.1 * len(chains),
            ),
        )

    # =====================================================
    # CONFIDENCE SCORING
    # =====================================================

    def _confidence(self, freq):

        if freq >= 5:
            return 0.95

        if freq >= 3:
            return 0.85

        if freq >= 2:
            return 0.75

        return 0.60