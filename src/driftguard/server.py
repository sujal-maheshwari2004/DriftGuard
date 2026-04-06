from fastmcp import FastMCP

from driftguard.models.event import Event
from driftguard.graph.graph_store import GraphStore
from driftguard.graph.merge_engine import MergeEngine
from driftguard.graph.prune_engine import PruneEngine
from driftguard.storage.persistence import Persistence
from driftguard.retrieval.retrieval_engine import RetrievalEngine
from driftguard.logging_config import configure_logging, get_logger


logger = get_logger(__name__)


# =====================================================
# INITIALIZE MEMORY STACK
# =====================================================

logger.info("Initializing DriftGuard runtime components")

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
logger.info("DriftGuard graph loaded with stats=%s", graph_store.stats())


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

    logger.info(
        "Registering mistake action=%r feedback=%r outcome=%r",
        action,
        feedback,
        outcome,
    )

    event = Event(
        action=action,
        feedback=feedback,
        outcome=outcome,
    )

    graph_store.add_event(event)
    graph_store.save()
    logger.info("Mistake stored successfully; stats=%s", graph_store.stats())

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

    logger.info("Querying memory for context=%r", context)
    response = retrieval_engine.query(context)
    logger.info(
        "Query complete context=%r warnings=%d chains=%d confidence=%.2f",
        context,
        len(response.warnings),
        len(response.chains),
        response.confidence,
    )

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
    logger.info("Starting deep prune with stats=%s", before)
    prune_engine.deep_prune(graph_store.graph)
    graph_store.save()
    after = graph_store.stats()
    logger.info("Deep prune finished with stats=%s", after)

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

    stats = graph_store.stats()
    logger.debug("Graph stats requested: %s", stats)
    return stats


# =====================================================
# START SERVER
# =====================================================

def main():
    configure_logging()
    logger.info("Starting DriftGuard MCP server")
    mcp.run()


if __name__ == "__main__":
    main()
