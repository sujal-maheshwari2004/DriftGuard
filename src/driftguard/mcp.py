from fastmcp import FastMCP

from driftguard.config import DriftGuardSettings
from driftguard.logging_config import get_logger
from driftguard.runtime import DriftGuardRuntime, build_runtime


logger = get_logger(__name__)


def create_mcp_server(
    runtime: DriftGuardRuntime | None = None,
    *,
    settings: DriftGuardSettings | None = None,
) -> FastMCP:
    runtime = runtime or build_runtime(settings=settings)
    logger.info("Creating MCP server")

    mcp = FastMCP("DriftGuard")

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

        return runtime.register_mistake(
            action=action,
            feedback=feedback,
            outcome=outcome,
        )

    @mcp.tool()
    def register_success(
        action: str,
        feedback: str,
        outcome: str,
    ):
        """
        Register a causal success event into DriftGuard memory.

        Args:
            action:   What was done (e.g. "add more salt")
            feedback: The signal received (e.g. "well seasoned")
            outcome:  The result (e.g. "dish praised")
        """

        return runtime.register_success(
            action=action,
            feedback=feedback,
            outcome=outcome,
        )

    @mcp.tool()
    def query_memory(context: str):
        """
        Retrieve causal warnings and reinforcements from DriftGuard memory.

        Args:
            context: The current action or situation being considered.
        """

        response = runtime.query_memory(context)

        return {
            "query": response.query,
            "warnings": [
                {
                    "trigger": warning.trigger,
                    "risk": warning.risk,
                    "frequency": warning.frequency,
                    "confidence": warning.confidence,
                }
                for warning in response.warnings
            ],
            "reinforcements": [
                {
                    "trigger": reinforcement.trigger,
                    "recommendation": reinforcement.recommendation,
                    "frequency": reinforcement.frequency,
                    "confidence": reinforcement.confidence,
                }
                for reinforcement in response.reinforcements
            ],
            "chains": response.chains,
            "confidence": response.confidence,
        }

    @mcp.tool()
    def deep_prune():
        """
        Run a full graph cleanup pass. This permanently deletes memories.

        Removes links that were never reinforced, nodes that have not been
        seen inside the stale window, and nodes left with no links. Anything
        newer than the stale window is kept. Call occasionally, not routinely.
        """

        return runtime.deep_prune()

    @mcp.tool()
    def graph_stats():
        """
        Return current memory graph statistics.
        """

        return runtime.graph_stats()

    @mcp.tool()
    def guard_metrics():
        """
        Return DriftGuard runtime metrics and review counters.
        """

        return runtime.metrics_snapshot()

    return mcp
