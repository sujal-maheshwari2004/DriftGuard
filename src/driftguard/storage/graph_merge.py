"""
Union one graph into another.

Every backend used to write by replacing whatever was already stored with the
caller's whole in-memory graph. Two runtimes pointed at the same file therefore
clobbered each other: the second save dropped every node the first had written,
with no error and no warning.

Saves are now read-modify-write, and this is the modify step.
"""

import networkx as nx

from driftguard.logging_config import get_logger
from driftguard.utils.node_roles import parse_roles


logger = get_logger(__name__)


def merge_graphs(base: nx.DiGraph, incoming: nx.DiGraph) -> nx.DiGraph:
    """
    Fold `incoming` into `base` and return `base`, which is mutated in place.

    Frequencies take the larger of the two rather than the sum. A runtime
    rewrites its entire graph on every save, so summing would re-add the same
    counts on each pass and inflate without bound. `max` never overcounts; it
    can undercount when two runtimes reinforce the same link independently,
    which is the safer direction for a value that drives warning confidence.
    """

    for node, incoming_data in incoming.nodes(data=True):
        if node not in base:
            base.add_node(node, **incoming_data)
            continue

        _merge_attributes(base.nodes[node], incoming_data, "first_seen", "last_seen")

    for src, dst, incoming_data in incoming.edges(data=True):
        if not base.has_edge(src, dst):
            base.add_edge(src, dst, **incoming_data)
            continue

        _merge_attributes(base[src][dst], incoming_data, "created_at", None)
        existing = base[src][dst]
        existing["weight"] = max(
            float(existing.get("weight", 1.0)),
            float(incoming_data.get("weight", 1.0)),
        )

    logger.debug(
        "Merged graph into base nodes=%d edges=%d",
        base.number_of_nodes(),
        base.number_of_edges(),
    )
    return base


def _merge_attributes(existing: dict, incoming: dict, oldest_key, newest_key) -> None:
    existing["frequency"] = max(
        int(existing.get("frequency", 1)),
        int(incoming.get("frequency", 1)),
    )

    if "type" in existing or "type" in incoming:
        roles = list(parse_roles(existing.get("type")))
        roles += [
            role for role in parse_roles(incoming.get("type")) if role not in roles
        ]
        existing["type"] = tuple(roles)

    # An embedding is a function of the node text, so either copy is fine —
    # but a stored None should never win over a real vector.
    if existing.get("embedding") is None and incoming.get("embedding") is not None:
        existing["embedding"] = incoming["embedding"]

    _keep(existing, incoming, oldest_key, min)
    _keep(existing, incoming, newest_key, max)


def _keep(existing: dict, incoming: dict, key, pick) -> None:
    if key is None:
        return

    values = [
        value
        for value in (existing.get(key), incoming.get(key))
        if value is not None
    ]

    if values:
        existing[key] = pick(values)
