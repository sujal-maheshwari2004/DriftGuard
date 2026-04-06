from fastmcp import FastMCP

from DriftGuard.models.event import Event
from DriftGuard.graph.graph_store import GraphStore
from DriftGuard.graph.merge_engine import MergeEngine
from DriftGuard.graph.prune_engine import PruneEngine
from DriftGuard.storage.persistence import Persistence
from DriftGuard.retrieval.retrieval_engine import RetrievalEngine


# =====================================================
# INITIALIZE MEMORY STACK
# =====================================================

merge_engine = MergeEngine()
prune_engine = PruneEngine()
persistence = Persistence()

graph_store = GraphStore(
    merge_engine=merge_engine,
    prune_engine=prune_engine,
    persistence_engine=persistence,
)

retrieval_engine = RetrievalEngine(graph_store)

graph_store.load()


# =====================================================
# CREATE MCP SERVER
# =====================================================

mcp = FastMCP("DriftGuard")


# =====================================================
# TOOL: REGISTER MISTAKE
# =====================================================

@mcp.tool()
def register_mistake(
    action: str,
    feedback: str,
    outcome: str,
):
    """
    Register a causal mistake event into DriftGuard memory.

    Args:
        action:   What was done (e.g. "increase salt")
        feedback: The signal received (e.g. "too salty")
        outcome:  The result (e.g. "dish ruined")
    """

    event = Event(
        action=action,
        feedback=feedback,
        outcome=outcome,
    )

    graph_store.add_event(event)
    graph_store.save()

    return {
        "status": "stored",
        "action": action,
        "feedback": feedback,
        "outcome": outcome,
    }


# =====================================================
# TOOL: QUERY MEMORY
# =====================================================

@mcp.tool()
def query_memory(context: str):
    """
    Retrieve causal warnings from DriftGuard memory.

    Args:
        context: The current action or situation being considered.
    """

    response = retrieval_engine.query(context)

    return {
        "query": response.query,
        "warnings": [
            {
                "trigger": w.trigger,
                "risk": w.risk,
                "frequency": w.frequency,
                "confidence": w.confidence,
            }
            for w in response.warnings
        ],
        "chains": response.chains,
        "confidence": response.confidence,
    }


# =====================================================
# TOOL: DEEP PRUNE (maintenance)
# =====================================================

@mcp.tool()
def deep_prune():
    """
    Run a full graph cleanup pass.

    Removes weak edges, stale nodes, and isolated nodes.
    Call occasionally to keep the memory graph healthy.
    """

    before = graph_store.stats()
    prune_engine.deep_prune(graph_store.graph)
    graph_store.save()
    after = graph_store.stats()

    return {
        "status": "pruned",
        "before": before,
        "after": after,
    }


# =====================================================
# TOOL: GRAPH STATS
# =====================================================

@mcp.tool()
def graph_stats():
    """
    Return current memory graph statistics.
    """

    return graph_store.stats()


# =====================================================
# START SERVER
# =====================================================

if __name__ == "__main__":
    mcp.run()