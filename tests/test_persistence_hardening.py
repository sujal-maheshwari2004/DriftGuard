import json
from datetime import datetime, UTC
from pathlib import Path
from uuid import uuid4

import networkx as nx
import numpy as np
import pytest

from driftguard.storage.persistence import (
    PERSISTENCE_FORMAT_NAME,
    PERSISTENCE_FORMAT_VERSION,
    Persistence,
)


def _build_graph() -> nx.DiGraph:
    graph = nx.DiGraph()
    now = datetime.now(UTC)
    graph.add_node(
        "test_node",
        type="action",
        embedding=np.array([1.0, 0.0], dtype=np.float32),
        frequency=3,
        first_seen=now,
        last_seen=now,
    )
    graph.add_edge(
        "test_node",
        "too salty",
        frequency=2,
        created_at=now,
    )
    return graph


@pytest.fixture
def writable_filepath():
    filepath = Path.cwd() / f"persistence-test-{uuid4().hex}.json"
    temp_path = filepath.with_suffix(f"{filepath.suffix}.tmp")

    yield filepath

    for path in (filepath, temp_path):
        if path.exists():
            path.unlink()


def test_persistence_saves_versioned_payload_atomically(writable_filepath):
    """Persistence should save a versioned wrapper and clean up the temp file."""

    filepath = writable_filepath
    persistence = Persistence(filepath=str(filepath))

    persistence.save_graph(_build_graph())

    with open(filepath, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    assert payload["format"] == PERSISTENCE_FORMAT_NAME
    assert payload["format_version"] == PERSISTENCE_FORMAT_VERSION
    assert "graph" in payload
    assert not filepath.with_suffix(".json.tmp").exists()


def test_persistence_loads_legacy_node_link_payload(writable_filepath):
    """Persistence should remain backward-compatible with legacy node-link JSON."""

    filepath = writable_filepath
    persistence = Persistence(filepath=str(filepath))
    persistence.save_graph(_build_graph())

    with open(filepath, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    with open(filepath, "w", encoding="utf-8") as handle:
        json.dump(payload["graph"], handle)

    loaded = persistence.load_graph()

    assert "test_node" in loaded.nodes
    assert loaded.nodes["test_node"]["frequency"] == 3


def test_persistence_rejects_unknown_format_version(writable_filepath):
    """Persistence should fail clearly for unsupported future format versions."""

    filepath = writable_filepath
    payload = {
        "format": PERSISTENCE_FORMAT_NAME,
        "format_version": 999,
        "graph": {"nodes": [], "links": []},
    }

    with open(filepath, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)

    persistence = Persistence(filepath=str(filepath))

    with pytest.raises(ValueError, match="Unsupported persistence format version"):
        persistence.load_graph()


def test_persistence_rejects_invalid_graph_payload(writable_filepath):
    """Persistence should fail clearly when the wrapped graph payload is malformed."""

    filepath = writable_filepath
    payload = {
        "format": PERSISTENCE_FORMAT_NAME,
        "format_version": PERSISTENCE_FORMAT_VERSION,
        "graph": {"nodes": []},
    }

    with open(filepath, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)

    persistence = Persistence(filepath=str(filepath))

    with pytest.raises(ValueError, match="invalid or incomplete"):
        persistence.load_graph()
