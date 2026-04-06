from datetime import datetime, UTC, timedelta


class PruneEngine:
    """
    Handles graph hygiene operations.

    Two modes:
    - light_prune: intentional no-op — called after every insert,
                   reserved for cheap future checks (e.g. cap on node count).
                   Do not add heavy operations here.
    - deep_prune:  full cleanup — call on a schedule or manual trigger.

    Design goals:
    - Never delete fresh knowledge
    - Keep repeated signals
    - Maintain retrieval performance
    """

    def __init__(
        self,
        node_stale_days: int = 60,
        edge_min_frequency: int = 2,
    ):
        """
        Parameters
        ----------
        node_stale_days:
            Nodes not updated within this window become eligible for removal.

        edge_min_frequency:
            Edges seen fewer times than this during deep_prune are removed.
        """

        self.node_stale_days = node_stale_days
        self.edge_min_frequency = edge_min_frequency

    # =====================================================
    # LIGHT PRUNE
    # =====================================================

    def light_prune(self, graph):
        """
        Called after every insertion. Intentionally minimal.

        Placeholder for cheap guards (e.g. hard node-count cap).
        Heavy operations belong in deep_prune.
        """

        pass

    # =====================================================
    # DEEP PRUNE
    # =====================================================

    def deep_prune(self, graph):
        """
        Full cleanup. Run on a schedule or via manual trigger.

        Order matters:
        1. Remove weak edges first
        2. Then stale nodes (may expose isolates)
        3. Then isolated nodes
        """

        self._remove_weak_edges(graph)
        self._remove_stale_nodes(graph)
        self._remove_isolated_nodes(graph)

    # =====================================================
    # REMOVE WEAK EDGES
    # =====================================================

    def _remove_weak_edges(self, graph):
        """
        Remove causal links that were never reinforced.
        """

        weak = [
            (src, dst)
            for src, dst, data in graph.edges(data=True)
            if data.get("frequency", 1) < self.edge_min_frequency
        ]

        for edge in weak:
            graph.remove_edge(*edge)

    # =====================================================
    # REMOVE STALE NODES
    # =====================================================

    def _remove_stale_nodes(self, graph):
        """
        Remove nodes not seen within the stale window.
        """

        now = datetime.now(UTC)
        cutoff = timedelta(days=self.node_stale_days)

        stale = [
            node
            for node in graph.nodes
            if (last := graph.nodes[node].get("last_seen"))
            and (now - last) > cutoff
        ]

        for node in stale:
            graph.remove_node(node)

    # =====================================================
    # REMOVE ISOLATED NODES
    # =====================================================

    def _remove_isolated_nodes(self, graph):
        """
        Remove nodes with no incoming or outgoing edges.
        """

        isolated = [
            node
            for node in graph.nodes
            if graph.in_degree(node) == 0 and graph.out_degree(node) == 0
        ]

        for node in isolated:
            graph.remove_node(node)