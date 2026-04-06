from dataclasses import dataclass

from driftguard.config import DEFAULT_SETTINGS, DriftGuardSettings
from driftguard.graph.graph_store import GraphStore
from driftguard.graph.merge_engine import MergeEngine
from driftguard.graph.prune_engine import PruneEngine
from driftguard.logging_config import get_logger
from driftguard.models.event import Event
from driftguard.retrieval.retrieval_engine import RetrievalEngine
from driftguard.storage.persistence import Persistence


logger = get_logger(__name__)


@dataclass
class DriftGuardRuntime:
    settings: DriftGuardSettings
    merge_engine: MergeEngine
    prune_engine: PruneEngine
    persistence: Persistence
    graph_store: GraphStore
    retrieval_engine: RetrievalEngine

    def register_mistake(
        self,
        action: str,
        feedback: str,
        outcome: str,
    ) -> dict:
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

        self.graph_store.add_event(event)
        self.graph_store.save()

        response = {
            "status": "stored",
            "action": action,
            "feedback": feedback,
            "outcome": outcome,
        }
        logger.info(
            "Mistake stored successfully; stats=%s",
            self.graph_store.stats(),
        )
        return response

    def query_memory(self, context: str):
        logger.info("Querying memory for context=%r", context)
        response = self.retrieval_engine.query(context)
        logger.info(
            "Query complete context=%r warnings=%d chains=%d confidence=%.2f",
            context,
            len(response.warnings),
            len(response.chains),
            response.confidence,
        )
        return response

    def deep_prune(self) -> dict:
        before = self.graph_store.stats()
        logger.info("Starting deep prune with stats=%s", before)
        self.prune_engine.deep_prune(self.graph_store.graph)
        self.graph_store.save()
        after = self.graph_store.stats()
        logger.info("Deep prune finished with stats=%s", after)
        return {
            "status": "pruned",
            "before": before,
            "after": after,
        }

    def graph_stats(self) -> dict:
        stats = self.graph_store.stats()
        logger.debug("Graph stats requested: %s", stats)
        return stats


def build_runtime(
    *,
    settings: DriftGuardSettings | None = None,
    merge_engine: MergeEngine | None = None,
    prune_engine: PruneEngine | None = None,
    persistence: Persistence | None = None,
    auto_load: bool = True,
) -> DriftGuardRuntime:
    logger.info("Building DriftGuard runtime")

    settings = settings or DEFAULT_SETTINGS

    merge_engine = merge_engine or MergeEngine(settings=settings)
    prune_engine = prune_engine or PruneEngine(
        node_stale_days=settings.prune_node_stale_days,
        edge_min_frequency=settings.prune_edge_min_frequency,
    )
    persistence = persistence or Persistence(filepath=settings.graph_filepath)

    graph_store = GraphStore(
        merge_engine=merge_engine,
        prune_engine=prune_engine,
        persistence_engine=persistence,
    )
    retrieval_engine = RetrievalEngine(
        graph_store,
        top_k=settings.retrieval_top_k,
        min_similarity=settings.retrieval_min_similarity,
    )

    runtime = DriftGuardRuntime(
        settings=settings,
        merge_engine=merge_engine,
        prune_engine=prune_engine,
        persistence=persistence,
        graph_store=graph_store,
        retrieval_engine=retrieval_engine,
    )

    if auto_load:
        runtime.graph_store.load()
        logger.info(
            "Runtime ready with loaded graph stats=%s",
            runtime.graph_store.stats(),
        )
    else:
        logger.info("Runtime ready without auto-loading persisted graph")

    return runtime
