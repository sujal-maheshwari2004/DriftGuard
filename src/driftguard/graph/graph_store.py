import networkx as nx

from datetime import datetime, UTC

from driftguard.logging_config import get_logger


logger = get_logger(__name__)


class GraphStore:
    """
    Central graph memory for DriftGuard.

    Stores causal chains as a directed graph:
        action → feedback → outcome

    Delegates to:
    - MergeEngine   for semantic deduplication
    - PruneEngine   for graph hygiene
    - Persistence   for disk I/O
    """

    def __init__(
        self,
        merge_engine,
        prune_engine,
        persistence_engine,
        metrics=None,
        traversal_max_depth: int = 3,
        traversal_max_branching: int = 10,
        traversal_max_paths: int = 100,
    ):
        self.graph = nx.DiGraph()
        self.merge_engine = merge_engine
        self.prune_engine = prune_engine
        self.persistence_engine = persistence_engine
        self.metrics = metrics
        self.traversal_max_depth = traversal_max_depth
        self.traversal_max_branching = traversal_max_branching
        self.traversal_max_paths = traversal_max_paths

    # =====================================================
    # PERSISTENCE
    # =====================================================

    def save(self):
        """Persist current graph to disk."""

        logger.debug("Saving graph with stats=%s", self.stats())
        self.persistence_engine.save_graph(self.graph)

    def load(self):
        """Load graph from disk. No-op if no file exists."""

        loaded = self.persistence_engine.load_graph()

        if loaded is not None:
            self.graph = loaded
            logger.info("Loaded graph from persistence with stats=%s", self.stats())
        else:
            logger.info("No persisted graph found; starting with an empty graph")

    # =====================================================
    # ADD EVENT
    # =====================================================

    def add_event(self, event):
        before = self.stats()

        action = self._get_or_create_node(event.action, "action")
        feedback = self._get_or_create_node(event.feedback, "feedback")
        outcome = self._get_or_create_node(event.outcome, "outcome")

        self._add_edge(action, feedback)
        self._add_edge(feedback, outcome)

        # light_prune is intentionally minimal — only run deep_prune on schedule
        self.prune_engine.light_prune(self.graph)
        logger.info(
            "Added event action=%r feedback=%r outcome=%r stats_before=%s stats_after=%s",
            action,
            feedback,
            outcome,
            before,
            self.stats(),
        )

    # =====================================================
    # FIND SIMILAR NODES
    # =====================================================

    def find_similar_nodes(
        self,
        text: str,
        node_type: str = None,
        top_k: int = 5,
        min_similarity: float = 0.0,
        include_scores: bool = False,
    ):
        matches = self.merge_engine.find_top_k_similar(
            text,
            self.graph,
            node_type,
            top_k,
            min_similarity,
            include_scores,
        )
        logger.debug(
            "Similarity search text=%r node_type=%r top_k=%d matches=%d min_similarity=%.2f include_scores=%s",
            text,
            node_type,
            top_k,
            len(matches),
            min_similarity,
            include_scores,
        )
        return matches

    # =====================================================
    # GET RELATED CHAINS
    # =====================================================

    def get_related_chains(
        self,
        node_text: str,
        depth: int | None = None,
        max_branching: int | None = None,
        max_paths: int | None = None,
    ):

        if node_text not in self.graph:
            return []

        depth = self.traversal_max_depth if depth is None else depth
        max_branching = (
            self.traversal_max_branching
            if max_branching is None
            else max_branching
        )
        max_paths = self.traversal_max_paths if max_paths is None else max_paths

        paths = []

        def dfs(node, path, remaining):
            if len(paths) >= max_paths:
                logger.debug(
                    "Stopping chain traversal for node=%r because max_paths=%d was reached",
                    node_text,
                    max_paths,
                )
                return

            if remaining == 0:
                paths.append(path)
                return

            neighbors = sorted(
                self.graph.successors(node),
                key=lambda neighbor: self.graph[node][neighbor].get(
                    "frequency", 1
                ),
                reverse=True,
            )
            neighbors = [neighbor for neighbor in neighbors if neighbor not in path]

            if not neighbors:
                paths.append(path)
                return

            for neighbor in neighbors[:max_branching]:
                dfs(neighbor, path + [neighbor], remaining - 1)
                if len(paths) >= max_paths:
                    break

        dfs(node_text, [node_text], depth)

        return paths

    # =====================================================
    # NODE ACCESS
    # =====================================================

    def get_node(self, node: str) -> dict:
        return dict(self.graph.nodes[node])

    # =====================================================
    # STATS
    # =====================================================

    def stats(self) -> dict:
        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
        }

    # =====================================================
    # INTERNAL: GET OR CREATE NODE
    # =====================================================

    def _get_or_create_node(self, text: str, node_type: str) -> str:

        normalized = self.merge_engine.normalize(text)

        existing = self.merge_engine.find_similar_node(
            normalized,
            node_type,
            self.graph,
        )

        if existing:
            node = self.graph.nodes[existing]
            node["frequency"] += 1
            node["last_seen"] = datetime.now(UTC)
            if self.metrics is not None:
                self.metrics.record_node_merged()
            logger.debug(
                "Merged %r into existing %s node=%r frequency=%d",
                text,
                node_type,
                existing,
                node["frequency"],
            )
            return existing

        return self._create_node(normalized, node_type)

    # =====================================================
    # INTERNAL: CREATE NODE
    # =====================================================

    def _create_node(self, text: str, node_type: str) -> str:

        embedding = self.merge_engine.embed(text)
        now = datetime.now(UTC)

        self.graph.add_node(
            text,
            type=node_type,
            embedding=embedding,
            frequency=1,
            first_seen=now,
            last_seen=now,
        )
        if self.metrics is not None:
            self.metrics.record_node_created()

        logger.debug("Created new %s node=%r", node_type, text)
        return text

    # =====================================================
    # INTERNAL: ADD EDGE
    # =====================================================

    def _add_edge(self, src: str, dst: str):

        if self.graph.has_edge(src, dst):
            self.graph[src][dst]["frequency"] += 1
            if self.metrics is not None:
                self.metrics.record_edge_reused()
            logger.debug(
                "Incremented edge %r -> %r frequency=%d",
                src,
                dst,
                self.graph[src][dst]["frequency"],
            )

        else:
            self.graph.add_edge(
                src,
                dst,
                frequency=1,
                weight=1.0,
                created_at=datetime.now(UTC),
            )
            if self.metrics is not None:
                self.metrics.record_edge_created()
            logger.debug("Created edge %r -> %r", src, dst)
