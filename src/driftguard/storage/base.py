from __future__ import annotations

from typing import Protocol

import networkx as nx


class GraphPersistence(Protocol):
    def save_graph(self, graph: nx.DiGraph, *, merge: bool = True) -> None:
        """
        Persist the graph.

        With merge=True (the default) the stored graph is read back and the
        caller's graph is folded into it, so a concurrent writer's memories
        survive. Pass merge=False to replace what is stored — deep_prune is
        the only caller that means to delete, and a merging save would
        resurrect everything it removed.
        """

    def load_graph(self) -> nx.DiGraph | None:
        """Load the full graph, or return None when no persisted data exists."""
