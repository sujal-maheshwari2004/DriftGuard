import pytest

from driftguard.evaluation import (
    MergeBenchmarkCase,
    RetrievalBenchmarkCase,
    evaluate_benchmark_suite,
    evaluate_merge_cases,
    evaluate_retrieval_cases,
    metric_summary,
)
from driftguard.graph.graph_store import GraphStore
from driftguard.graph.merge_engine import MergeEngine
from driftguard.graph.prune_engine import PruneEngine
from driftguard.models.event import Event
from driftguard.retrieval.retrieval_engine import RetrievalEngine


class DummyPersistence:
    def save_graph(self, graph):
        pass

    def load_graph(self):
        return None


class StubEmbeddingEngine:
    def __init__(self, embeddings: dict[str, list[float]]):
        self.embeddings = embeddings

    def embed(self, text: str):
        return self.embeddings[text]

    def model_name(self) -> str:
        return "stub"


@pytest.fixture
def benchmark_stack(monkeypatch):
    monkeypatch.setattr(
        "driftguard.graph.merge_engine.normalize_text",
        lambda text: text.lower().strip(),
    )
    monkeypatch.setattr(
        "driftguard.graph.merge_engine.EmbeddingEngine",
        lambda model_name=None, device=None: StubEmbeddingEngine(
            {
                "increase salt": [1.0, 0.0, 0.0],
                "add more salt": [0.97, 0.03, 0.0],
                "raise heat": [0.0, 1.0, 0.0],
                "too salty": [0.0, 0.0, 1.0],
                "dish ruined": [0.0, 0.0, 1.0],
            }
        ),
    )

    merge_engine = MergeEngine()
    graph_store = GraphStore(
        merge_engine=merge_engine,
        prune_engine=PruneEngine(),
        persistence_engine=DummyPersistence(),
    )
    graph_store.add_event(
        Event(
            action="increase salt",
            feedback="too salty",
            outcome="dish ruined",
        )
    )
    retrieval_engine = RetrievalEngine(
        graph_store,
        min_similarity=0.60,
    )
    return merge_engine, graph_store, retrieval_engine


def test_metric_summary_handles_zero_denominators():
    """Metric summaries should remain well-defined for empty benchmarks."""

    summary = metric_summary(0, 0, 0)

    assert summary.precision == 0.0
    assert summary.recall == 0.0
    assert summary.f1 == 0.0


def test_evaluate_merge_cases_reports_precision_and_recall(benchmark_stack):
    """Merge benchmarks should score paraphrase merges and reject unrelated actions."""

    merge_engine, graph_store, _ = benchmark_stack

    metrics, results = evaluate_merge_cases(
        merge_engine,
        graph_store.graph,
        (
            MergeBenchmarkCase(
                name="seasoning paraphrase",
                query="add more salt",
                node_type="action",
                expected_anchor="increase salt",
            ),
            MergeBenchmarkCase(
                name="unrelated action",
                query="raise heat",
                node_type="action",
                expected_anchor=None,
            ),
        ),
    )

    assert metrics.true_positive == 1
    assert metrics.false_positive == 0
    assert metrics.false_negative == 0
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert all(result.passed for result in results)


def test_evaluate_retrieval_cases_reports_expected_risks(benchmark_stack):
    """Retrieval benchmarks should measure expected warnings across queries."""

    _, _, retrieval_engine = benchmark_stack

    metrics, results = evaluate_retrieval_cases(
        retrieval_engine,
        (
            RetrievalBenchmarkCase(
                name="known salt risk",
                query="add more salt",
                expected_risks=("too salty",),
            ),
            RetrievalBenchmarkCase(
                name="unrelated heat query",
                query="raise heat",
                expected_risks=(),
            ),
        ),
    )

    assert metrics.true_positive == 1
    assert metrics.false_positive == 0
    assert metrics.false_negative == 0
    assert metrics.f1 == 1.0
    assert results[0].predicted_risks == ("too salty",)
    assert results[1].predicted_risks == ()


def test_evaluate_benchmark_suite_returns_combined_report(benchmark_stack):
    """Benchmark suite reports should combine merge and retrieval evaluations."""

    merge_engine, graph_store, retrieval_engine = benchmark_stack

    report = evaluate_benchmark_suite(
        merge_engine=merge_engine,
        graph=graph_store.graph,
        retrieval_engine=retrieval_engine,
        merge_cases=(
            MergeBenchmarkCase(
                name="seasoning paraphrase",
                query="add more salt",
                node_type="action",
                expected_anchor="increase salt",
            ),
        ),
        retrieval_cases=(
            RetrievalBenchmarkCase(
                name="known salt risk",
                query="add more salt",
                expected_risks=("too salty",),
            ),
        ),
    )

    assert report.merge_metrics.precision == 1.0
    assert report.retrieval_metrics.recall == 1.0
    assert report.merge_results[0].passed is True
    assert report.retrieval_results[0].passed is True
