import json
import numpy as np
import networkx as nx
import os

from pathlib import Path
from datetime import datetime

from driftguard.errors import PersistenceError
from driftguard.storage.embedding_codec import decode as decode_embedding
from driftguard.storage.embedding_codec import encode as encode_embedding
from driftguard.logging_config import get_logger
from driftguard.storage.file_lock import FileLock
from driftguard.storage.graph_merge import merge_graphs


logger = get_logger(__name__)
PERSISTENCE_FORMAT_NAME = "driftguard_graph"
PERSISTENCE_FORMAT_VERSION = 2
# Version 1 wrote embeddings as JSON lists of floats; version 2 packs them as
# base64 float32. Both are readable.
SUPPORTED_FORMAT_VERSIONS = (1, 2)
NODE_LINK_EDGES_KEY = "edges"


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
            return encode_embedding(obj)

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

    def save_graph(self, graph: nx.DiGraph, *, merge: bool = True):

        self.filepath.parent.mkdir(parents=True, exist_ok=True)

        with FileLock(self.filepath):
            self._write_locked(graph, merge=merge)

    def _write_locked(self, graph: nx.DiGraph, *, merge: bool):

        if merge:
            stored = self.load_graph()

            if stored is not None:
                graph = merge_graphs(stored, graph)

        payload = {
            "format": PERSISTENCE_FORMAT_NAME,
            "format_version": PERSISTENCE_FORMAT_VERSION,
            # networkx renamed this key from "links" to "edges" in 3.6. Pin it
            # so a file written by one version is readable by the other.
            "graph": nx.node_link_data(graph, edges=NODE_LINK_EDGES_KEY),
        }
        temp_path = self.filepath.with_suffix(f"{self.filepath.suffix}.tmp")

        try:
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, cls=_GraphEncoder, indent=2)
            os.replace(temp_path, self.filepath)
        finally:
            if temp_path.exists():
                temp_path.unlink()

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

        try:
            with open(self.filepath, "r", encoding="utf-8") as f:
                raw_payload = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            raise PersistenceError(
                f"DriftGuard could not read the graph at {self.filepath}: {exc}. "
                f"Move or delete the file to start from an empty graph."
            ) from exc

        graph_data = self._extract_graph_data(raw_payload)
        # Files written before the key was pinned may use either name.
        edges_key = NODE_LINK_EDGES_KEY if "edges" in graph_data else "links"
        graph = nx.node_link_graph(graph_data, directed=True, edges=edges_key)

        # Restore numpy arrays and datetime objects
        for node in graph.nodes:
            node_data = graph.nodes[node]

            if node_data.get("embedding") is not None:
                node_data["embedding"] = decode_embedding(node_data["embedding"])

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

    def _extract_graph_data(self, raw_payload: dict) -> dict:
        if not isinstance(raw_payload, dict):
            raise self._unreadable("payload is not a JSON object")

        # Backward compatibility for the original node-link JSON format.
        if (
            "format" not in raw_payload
            and "format_version" not in raw_payload
            and self._looks_like_node_link_graph(raw_payload)
        ):
            logger.warning(
                "Loading legacy persistence format from %s without version metadata",
                self.filepath,
            )
            return raw_payload

        payload_format = raw_payload.get("format")
        version = raw_payload.get("format_version")
        graph_data = raw_payload.get("graph")

        if payload_format != PERSISTENCE_FORMAT_NAME:
            raise self._unreadable(f"unsupported format {payload_format!r}")

        if version not in SUPPORTED_FORMAT_VERSIONS:
            raise self._unreadable(f"unsupported format version {version!r}")

        if not self._looks_like_node_link_graph(graph_data):
            raise self._unreadable("graph payload is invalid or incomplete")

        return graph_data

    def _unreadable(self, reason: str) -> PersistenceError:
        """
        Every load failure names the file and how to recover from it.

        The graph is loaded during construction, so an unreadable file used to
        abort startup with a bare ValueError or JSONDecodeError that callers
        could not tell apart from any other bad value.
        """

        return PersistenceError(
            f"DriftGuard could not read the graph at {self.filepath}: {reason}. "
            f"Move or delete the file to start from an empty graph."
        )

    def _looks_like_node_link_graph(self, payload: dict | None) -> bool:
        return (
            isinstance(payload, dict)
            and "nodes" in payload
            and ("links" in payload or "edges" in payload)
        )
