from dataclasses import dataclass


@dataclass(frozen=True)
class DriftGuardSettings:
    graph_filepath: str = "driftguard_graph.json"
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_device: str | None = None
    retrieval_top_k: int = 5
    similarity_threshold_action: float = 0.72
    similarity_threshold_feedback: float = 0.70
    similarity_threshold_outcome: float = 0.88
    prune_node_stale_days: int = 60
    prune_edge_min_frequency: int = 2
    log_level: str = "INFO"

    def threshold_for(self, node_type: str) -> float:
        return {
            "action": self.similarity_threshold_action,
            "feedback": self.similarity_threshold_feedback,
            "outcome": self.similarity_threshold_outcome,
        }.get(node_type, 0.85)


DEFAULT_SETTINGS = DriftGuardSettings()

# Backwards-compatible module constants for callers that still import them.
SIM_THRESHOLD_ACTION = DEFAULT_SETTINGS.similarity_threshold_action
SIM_THRESHOLD_FEEDBACK = DEFAULT_SETTINGS.similarity_threshold_feedback
SIM_THRESHOLD_OUTCOME = DEFAULT_SETTINGS.similarity_threshold_outcome
