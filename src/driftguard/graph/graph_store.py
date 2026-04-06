import networkx as nx

from datetime import datetime, UTC


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
    ):
        self.graph = nx.DiGraph()
        self.merge_engine = merge_engine
        self.prune_engine = prune_engine
        self.persistence_engine = persistence_engine

    # =====================================================
    # PERSISTENCE
    # =====================================================

    def save(self):
        """Persist current graph to disk."""

        self.persistence_engine.save_graph(self.graph)

    def load(self):
        """Load graph from disk. No-op if no file exists."""

        loaded = self.persistence_engine.load_graph()

        if loaded is not None:
            self.graph = loaded

    # =====================================================
    # ADD EVENT
    # =====================================================

    def add_event(self, event):

        action = self._get_or_create_node(event.action, "action")
        feedback = self._get_or_create_node(event.feedback, "feedback")
        outcome = self._get_or_create_node(event.outcome, "outcome")

        self._add_edge(action, feedback)
        self._add_edge(feedback, outcome)

        # light_prune is intentionally minimal — only run deep_prune on schedule
        self.prune_engine.light_prune(self.graph)

    # =====================================================
    # FIND SIMILAR NODES
    # =====================================================

    def find_similar_nodes(
        self,
        text: str,
        node_type: str = None,
        top_k: int = 5,
    ):
        return self.merge_engine.find_top_k_similar(
            text,
            self.graph,
            node_type,
            top_k,
        )

    # =====================================================
    # GET RELATED CHAINS
    # =====================================================

    def get_related_chains(self, node_text: str, depth: int = 3):

        if node_text not in self.graph:
            return []

        paths = []

        def dfs(node, path, remaining):

            if remaining == 0:
                paths.append(path)
                return

            neighbors = list(self.graph.successors(node))

            if not neighbors:
                paths.append(path)
                return

            for neighbor in neighbors:
                dfs(neighbor, path + [neighbor], remaining - 1)

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

        return text

    # =====================================================
    # INTERNAL: ADD EDGE
    # =====================================================

    def _add_edge(self, src: str, dst: str):

        if self.graph.has_edge(src, dst):
            self.graph[src][dst]["frequency"] += 1

        else:
            self.graph.add_edge(
                src,
                dst,
                frequency=1,
                weight=1.0,
                created_at=datetime.now(UTC),
            )