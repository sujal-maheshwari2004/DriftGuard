from __future__ import annotations

from typing import Protocol

import networkx as nx


class GraphPersistence(Protocol):
    def save_graph(self, graph: nx.DiGraph) -> None:
        """Persist the full graph."""

    def load_graph(self) -> nx.DiGraph | None:
        """Load the full graph, or return None when no persisted data exists."""
