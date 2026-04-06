import json
import numpy as np
import networkx as nx

from pathlib import Path
from datetime import datetime

from driftguard.logging_config import get_logger


logger = get_logger(__name__)


# =====================================================
# CUSTOM JSON ENCODER
# =====================================================

class _GraphEncoder(json.JSONEncoder):
    """
    Handles types that standard json cannot serialize:
    - numpy arrays  → list
    - datetime      → ISO string
    """

    def default(self, obj):

        if isinstance(obj, np.ndarray):
            return obj.tolist()

        if isinstance(obj, datetime):
            return obj.isoformat()

        return super().default(obj)


# =====================================================
# PERSISTENCE ENGINE
# =====================================================

class Persistence:
    """
    Handles saving and loading DriftGuard graph memory.

    Uses JSON + networkx node_link format instead of pickle.

    Benefits over pickle:
    - Human-readable on disk
    - Safe across Python versions
    - No arbitrary code execution risk
    - Survives class renames
    """

    def __init__(self, filepath: str = "driftguard_graph.json"):

        self.filepath = Path(filepath)
        logger.info("Persistence configured with filepath=%s", self.filepath)

    # =====================================================
    # SAVE
    # =====================================================

    def save_graph(self, graph: nx.DiGraph):

        self.filepath.parent.mkdir(parents=True, exist_ok=True)

        data = nx.node_link_data(graph)

        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, cls=_GraphEncoder, indent=2)
        logger.info(
            "Saved graph to %s nodes=%d edges=%d",
            self.filepath,
            graph.number_of_nodes(),
            graph.number_of_edges(),
        )

    # =====================================================
    # LOAD
    # =====================================================

    def load_graph(self) -> nx.DiGraph | None:

        if not self.filepath.exists():
            logger.info("Persistence file does not exist at %s", self.filepath)
            return None

        with open(self.filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        graph = nx.node_link_graph(data, directed=True)

        # Restore numpy arrays and datetime objects
        for node in graph.nodes:
            node_data = graph.nodes[node]

            if "embedding" in node_data and isinstance(
                node_data["embedding"], list
            ):
                node_data["embedding"] = np.array(
                    node_data["embedding"], dtype=np.float32
                )

            for key in ("first_seen", "last_seen"):
                if key in node_data and isinstance(node_data[key], str):
                    node_data[key] = datetime.fromisoformat(node_data[key])

        for src, dst in graph.edges:
            edge_data = graph[src][dst]

            if "created_at" in edge_data and isinstance(
                edge_data["created_at"], str
            ):
                edge_data["created_at"] = datetime.fromisoformat(
                    edge_data["created_at"]
                )

        logger.info(
            "Loaded graph from %s nodes=%d edges=%d",
            self.filepath,
            graph.number_of_nodes(),
            graph.number_of_edges(),
        )
        return graph
