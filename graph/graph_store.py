import networkx as nx

from datetime import datetime, UTC
from typing import Optional, List, Dict, Any


class GraphStore:

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


    def add_event(self, event):
        action = self._get_or_create_node(
            event.action,
            "action",
        )

        feedback = self._get_or_create_node(
            event.feedback,
            "feedback",
        )

        outcome = self._get_or_create_node(
            event.outcome,
            "outcome",
        )

        self._add_edge(action, feedback)
        self._add_edge(feedback, outcome)
        self.prune_engine.light_prune(self.graph)


    def find_similar_nodes(
        self,
        text,
        node_type=None,
        top_k=5,
    ):

        return self.merge_engine.find_top_k_similar(
            text,
            self.graph,
            node_type,
            top_k,
        )


    def get_related_chains(
        self,
        node_text,
        depth=3,
    ):

        if node_text not in self.graph:
            return []

        paths = []

        def dfs(node, path, depth):

            if depth == 0:
                return

            neighbors = list(
                self.graph.successors(node)
            )

            if not neighbors:
                paths.append(path)
                return

            for n in neighbors:
                dfs(
                    n,
                    path + [n],
                    depth - 1,
                )
        dfs(node_text, [node_text], depth)

        return paths


    def get_node(self, node):
        return self.graph.nodes[node]


    def stats(self):
        return dict(
            nodes=self.graph.number_of_nodes(),
            edges=self.graph.number_of_edges(),
        )


    def _get_or_create_node(
        self,
        text,
        node_type,
    ):

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

        return self._create_node(
            normalized,
            node_type,
        )


    def _create_node(
        self,
        text,
        node_type,
    ):

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


    def _add_edge(
        self,
        src,
        dst,
    ):

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