from datetime import datetime, UTC, timedelta

from driftguard.logging_config import get_logger


logger = get_logger(__name__)


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
            Nodes and edges not updated within this window become eligible
            for removal. Nothing newer than this window is ever deleted.

        edge_min_frequency:
            Edges seen fewer times than this are removed by deep_prune, but
            only once they are also older than node_stale_days.
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

        logger.debug("Light prune invoked; no-op with current policy")

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

        before = {
            "nodes": graph.number_of_nodes(),
            "edges": graph.number_of_edges(),
        }

        removed_weak_edges = self._remove_weak_edges(graph)
        removed_stale_nodes = self._remove_stale_nodes(graph)
        removed_isolated_nodes = self._remove_isolated_nodes(graph)

        after = {
            "nodes": graph.number_of_nodes(),
            "edges": graph.number_of_edges(),
        }
        logger.info("Deep prune completed before=%s after=%s", before, after)
        return {
            "before": before,
            "after": after,
            "removed_weak_edges": removed_weak_edges,
            "removed_stale_nodes": removed_stale_nodes,
            "removed_isolated_nodes": removed_isolated_nodes,
        }

    # =====================================================
    # REMOVE WEAK EDGES
    # =====================================================

    def _remove_weak_edges(self, graph):
        """
        Remove causal links that were never reinforced *and* have gone stale.

        Frequency alone is not enough: with the default edge_min_frequency of
        2, every memory recorded exactly once is "weak" the moment it is
        written, so an unguarded pass deletes the whole graph. An edge only
        becomes eligible once it has also survived the stale window without
        being reinforced. Edges with no created_at (legacy files) are kept.
        """

        weak = [
            (src, dst)
            for src, dst, data in graph.edges(data=True)
            if data.get("frequency", 1) < self.edge_min_frequency
            and self._is_stale(data.get("created_at"))
        ]

        for edge in weak:
            graph.remove_edge(*edge)

        return len(weak)

    # =====================================================
    # REMOVE STALE NODES
    # =====================================================

    def _remove_stale_nodes(self, graph):
        """
        Remove nodes not seen within the stale window.
        """

        stale = [
            node
            for node in graph.nodes
            if self._is_stale(graph.nodes[node].get("last_seen"))
        ]

        for node in stale:
            graph.remove_node(node)

        return len(stale)

    # =====================================================
    # REMOVE ISOLATED NODES
    # =====================================================

    def _remove_isolated_nodes(self, graph):
        """
        Remove stale nodes left with no incoming or outgoing edges.

        Freshness wins over tidiness: a node that lost its edges but was seen
        recently is still live knowledge, so it stays until it ages out.
        """

        isolated = [
            node
            for node in graph.nodes
            if graph.in_degree(node) == 0
            and graph.out_degree(node) == 0
            and self._is_stale(graph.nodes[node].get("last_seen"))
        ]

        for node in isolated:
            graph.remove_node(node)

        return len(isolated)

    # =====================================================
    # STALENESS
    # =====================================================

    def _is_stale(self, timestamp) -> bool:
        """
        True when timestamp is older than the stale window.

        A missing or malformed timestamp is treated as not stale, so a graph
        written by an older version is never deleted on a guess.
        """

        if not isinstance(timestamp, datetime):
            return False

        return (datetime.now(UTC) - timestamp) > timedelta(days=self.node_stale_days)
