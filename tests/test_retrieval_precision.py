from dataclasses import dataclass

import networkx as nx
import pytest

from driftguard.config import DriftGuardSettings
from driftguard.graph.merge_engine import MergeEngine
from driftguard.retrieval.retrieval_engine import RetrievalEngine


class StubEmbeddingEngine:
    def __init__(self, embeddings: dict[str, list[float]]):
        self.embeddings = embeddings

    def embed(self, text: str):
        return self.embeddings[text]

    def model_name(self) -> str:
        return "stub"


def test_merge_engine_filters_weak_matches_and_can_return_scores():
    graph = nx.DiGraph()
    graph.add_node("strong", type="action", embedding=[1.0, 0.0])
    graph.add_node("medium", type="action", embedding=[0.8, 0.6])
    graph.add_node("weak", type="action", embedding=[0.2, 0.98])

    merge_engine = MergeEngine(
        settings=DriftGuardSettings(),
        embedding_engine=StubEmbeddingEngine({"query": [1.0, 0.0]}),
    )

    results = merge_engine.find_top_k_similar(
        "query",
        graph,
        node_type="action",
        top_k=3,
        min_similarity=0.75,
        include_scores=True,
    )

    assert results == [("strong", 1.0), ("medium", 0.8)]


@dataclass
class FakeNodeMatchStore:
    matches: list[tuple[str, float]]

    def __post_init__(self):
        self.graph = nx.DiGraph()
        self.graph.add_edge("increase salt", "too salty", frequency=1)
        self.find_calls = []

    def find_similar_nodes(
        self,
        text: str,
        node_type: str = None,
        top_k: int = 5,
        min_similarity: float = 0.0,
        include_scores: bool = False,
    ):
        self.find_calls.append(
            {
                "text": text,
                "node_type": node_type,
                "top_k": top_k,
                "min_similarity": min_similarity,
                "include_scores": include_scores,
            }
        )

        results = [
            match for match in self.matches[:top_k] if match[1] >= min_similarity
        ]
        return results if include_scores else [node for node, _ in results]

    def get_related_chains(self, node_text: str, depth: int = 3):
        if node_text == "increase salt":
            return [["increase salt", "too salty", "dish ruined"]]
        return []

    def get_node(self, node: str) -> dict:
        return {"frequency": 1}


def test_retrieval_engine_filters_out_weak_matches():
    store = FakeNodeMatchStore(matches=[("increase salt", 0.58)])
    retriever = RetrievalEngine(
        store,
        top_k=3,
        min_similarity=0.60,
    )

    result = retriever.query("increase salt")

    assert result.warnings == []
    assert result.chains == []
    assert result.confidence == 0.0
    assert store.find_calls == [
        {
            "text": "increase salt",
            "node_type": "action",
            "top_k": 3,
            "min_similarity": 0.60,
            "include_scores": True,
        }
    ]


def test_retrieval_confidence_increases_with_similarity_score():
    low_similarity_store = FakeNodeMatchStore(matches=[("increase salt", 0.65)])
    high_similarity_store = FakeNodeMatchStore(matches=[("increase salt", 0.95)])

    low_retriever = RetrievalEngine(
        low_similarity_store,
        min_similarity=0.60,
    )
    high_retriever = RetrievalEngine(
        high_similarity_store,
        min_similarity=0.60,
    )

    low_result = low_retriever.query("increase salt")
    high_result = high_retriever.query("increase salt")

    assert low_result.warnings
    assert high_result.warnings
    assert high_result.warnings[0].confidence > low_result.warnings[0].confidence
    assert high_result.confidence == high_result.warnings[0].confidence
    assert low_result.confidence == pytest.approx(0.6175)
    assert high_result.confidence == pytest.approx(0.7225)
