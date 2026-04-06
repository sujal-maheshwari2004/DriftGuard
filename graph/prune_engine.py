from datetime import datetime, UTC, timedelta


class PruneEngine:
    """
    Handles graph hygiene operations.

    Responsibilities:
    - remove weak edges (low-frequency)
    - remove stale nodes (not seen recently)
    - remove isolated nodes (no connections)

    Design goals:
    - never delete fresh knowledge
    - keep repeated signals
    - maintain retrieval performance
    """

    def __init__(
        self,
        node_stale_days=60,
        edge_min_frequency=2,
    ):
        """
        Parameters

        node_stale_days:
            Nodes not updated within this window
            become eligible for removal.

        edge_min_frequency:
            Edges with frequency lower than this
            threshold may be removed.
        """

        self.node_stale_days = node_stale_days
        self.edge_min_frequency = edge_min_frequency


    # =====================================================
    # LIGHT PRUNE
    # =====================================================

    def light_prune(self, graph):
        """
        Runs after every insertion.

        Cheap operations only.
        Keeps graph clean without damaging
        newly learned relationships.
        """

        # Currently safe to skip edge pruning here.
        # We preserve fresh edges until reinforcement happens.

        pass


    # =====================================================
    # DEEP PRUNE
    # =====================================================

    def deep_prune(self, graph):
        """
        Runs occasionally (manual trigger / scheduled job).

        Performs heavier cleanup.
        """

        self._remove_weak_edges(graph)

        self._remove_stale_nodes(graph)

        self._remove_isolated_nodes(graph)


    # =====================================================
    # REMOVE WEAK EDGES
    # =====================================================

    def _remove_weak_edges(self, graph):
        """
        Remove edges that never became reinforced.
        """

        weak_edges = []

        for src, dst, data in graph.edges(data=True):

            frequency = data.get("frequency", 1)

            if frequency < self.edge_min_frequency:

                weak_edges.append((src, dst))

        for edge in weak_edges:

            graph.remove_edge(*edge)


    # =====================================================
    # REMOVE STALE NODES
    # =====================================================

    def _remove_stale_nodes(self, graph):
        """
        Remove nodes not used recently.
        """

        now = datetime.now(UTC)

        stale_nodes = []

        for node in graph.nodes:

            last_seen = graph.nodes[node].get("last_seen")

            if not last_seen:
                continue

            age = now - last_seen

            if age > timedelta(days=self.node_stale_days):

                stale_nodes.append(node)

        for node in stale_nodes:

            graph.remove_node(node)


    # =====================================================
    # REMOVE ISOLATED NODES
    # =====================================================

    def _remove_isolated_nodes(self, graph):
        """
        Remove nodes with no edges.
        """

        isolated_nodes = []

        for node in graph.nodes:

            if (
                graph.in_degree(node) == 0
                and graph.out_degree(node) == 0
            ):

                isolated_nodes.append(node)

        for node in isolated_nodes:

            graph.remove_node(node)