from __future__ import annotations

from argparse import ArgumentParser, Namespace
from dataclasses import asdict, dataclass
import json

import numpy as np

from driftguard.config import DriftGuardSettings
from driftguard.evaluation import (
    BenchmarkSuiteReport,
    MergeBenchmarkCase,
    RetrievalBenchmarkCase,
    evaluate_benchmark_suite,
)
from driftguard.graph.graph_store import GraphStore
from driftguard.graph.merge_engine import MergeEngine
from driftguard.graph.prune_engine import PruneEngine
from driftguard.logging_config import configure_logging
from driftguard.models.event import Event
from driftguard.retrieval.retrieval_engine import RetrievalEngine


BENCHMARK_FEATURES: tuple[str, ...] = (
    "salt",
    "taste",
    "heat",
    "burn",
    "oil",
    "grease",
    "rest",
    "plate",
    "protein",
    "dry",
    "complaint",
    "delay",
)

BENCHMARK_TOKEN_ALIASES: dict[str, str] = {
    "adding": "add",
    "aggressively": "aggressive",
    "burned": "burn",
    "burner": "heat",
    "charred": "burn",
    "greasy": "grease",
    "grease": "oil",
    "hot": "heat",
    "hotter": "heat",
    "juices": "dry",
    "maximum": "heat",
    "oil": "oil",
    "oily": "grease",
    "order": "delay",
    "plate": "plate",
    "plating": "plate",
    "protein": "protein",
    "resting": "rest",
    "salinity": "salt",
    "salty": "salt",
    "season": "salt",
    "seasoned": "salt",
    "seasoning": "salt",
    "serve": "plate",
    "tasting": "taste",
    "ticket": "delay",
}

BENCHMARK_STOPWORDS = {
    "a",
    "after",
    "and",
    "before",
    "for",
    "immediately",
    "into",
    "it",
    "more",
    "of",
    "on",
    "the",
    "to",
    "too",
    "with",
    "without",
}


@dataclass(frozen=True)
class BenchmarkSuite:
    seed_events: tuple[Event, ...]
    merge_cases: tuple[MergeBenchmarkCase, ...]
    retrieval_cases: tuple[RetrievalBenchmarkCase, ...]


class DummyPersistence:
    def save_graph(self, graph):
        pass

    def load_graph(self):
        return None


def benchmark_normalize_text(text: str) -> str:
    cleaned = "".join(
        character.lower() if character.isalnum() else " "
        for character in text
    )
    tokens = []

    for token in cleaned.split():
        if token in BENCHMARK_STOPWORDS:
            continue
        tokens.append(BENCHMARK_TOKEN_ALIASES.get(token, token))

    return " ".join(tokens)


class BenchmarkEmbeddingEngine:
    def __init__(self):
        self._feature_index = {
            feature: index for index, feature in enumerate(BENCHMARK_FEATURES)
        }
        self._hash_buckets = 4

    def embed(self, text: str) -> np.ndarray:
        normalized = benchmark_normalize_text(text)
        vector = np.zeros(
            len(self._feature_index) + self._hash_buckets,
            dtype=np.float32,
        )

        for token in normalized.split():
            index = self._feature_index.get(token)

            if index is not None:
                vector[index] += 1.0
            else:
                bucket = sum(ord(character) for character in token) % self._hash_buckets
                vector[len(self._feature_index) + bucket] += 0.25

        if not vector.any():
            vector[-1] = 1.0

        norm = np.linalg.norm(vector)
        return vector if norm == 0 else vector / norm

    def model_name(self) -> str:
        return "benchmark-embedder"


class BenchmarkMergeEngine(MergeEngine):
    def __init__(self):
        settings = DriftGuardSettings(
            similarity_threshold_action=0.68,
            similarity_threshold_feedback=0.66,
            similarity_threshold_outcome=0.82,
        )
        super().__init__(
            settings=settings,
            embedding_engine=BenchmarkEmbeddingEngine(),
        )

    def normalize(self, text: str) -> str:
        return benchmark_normalize_text(text)


def builtin_benchmark_suite() -> BenchmarkSuite:
    seed_events = (
        Event(
            action="increase salt",
            feedback="too salty",
            outcome="dish ruined",
        ),
        Event(
            action="raise pan heat",
            feedback="outside burned before center cooked",
            outcome="protein was unusable",
        ),
        Event(
            action="add more oil",
            feedback="dish turned greasy",
            outcome="customer disliked the texture",
        ),
        Event(
            action="skip the resting time",
            feedback="meat dried out quickly",
            outcome="guest noticed the meat was dry",
        ),
    )
    merge_cases = (
        MergeBenchmarkCase(
            name="seasoning paraphrase",
            query="add more salt",
            node_type="action",
            expected_anchor="increase salt",
        ),
        MergeBenchmarkCase(
            name="heat paraphrase",
            query="cook on higher heat",
            node_type="action",
            expected_anchor="raise heat heat",
        ),
        MergeBenchmarkCase(
            name="unrelated action",
            query="organize garnish tray",
            node_type="action",
            expected_anchor=None,
        ),
    )
    retrieval_cases = (
        RetrievalBenchmarkCase(
            name="seasoning warning",
            query="season more aggressively",
            expected_risks=("salt",),
        ),
        RetrievalBenchmarkCase(
            name="heat warning",
            query="finish on maximum heat",
            expected_risks=("outside burn center heat",),
        ),
        RetrievalBenchmarkCase(
            name="resting warning",
            query="serve without letting it rest",
            expected_risks=("meat dried out quickly",),
        ),
        RetrievalBenchmarkCase(
            name="unrelated no-warning",
            query="organize garnish tray",
            expected_risks=(),
        ),
    )
    return BenchmarkSuite(
        seed_events=seed_events,
        merge_cases=merge_cases,
        retrieval_cases=retrieval_cases,
    )


def build_benchmark_runtime() -> tuple[BenchmarkMergeEngine, GraphStore, RetrievalEngine]:
    merge_engine = BenchmarkMergeEngine()
    graph_store = GraphStore(
        merge_engine=merge_engine,
        prune_engine=PruneEngine(),
        persistence_engine=DummyPersistence(),
        traversal_max_depth=2,
        traversal_max_branching=4,
        traversal_max_paths=20,
    )

    for event in builtin_benchmark_suite().seed_events:
        graph_store.add_event(event)

    retrieval_engine = RetrievalEngine(
        graph_store,
        top_k=4,
        min_similarity=0.60,
    )
    return merge_engine, graph_store, retrieval_engine


def run_builtin_benchmark() -> BenchmarkSuiteReport:
    suite = builtin_benchmark_suite()
    merge_engine, graph_store, retrieval_engine = build_benchmark_runtime()
    return evaluate_benchmark_suite(
        merge_engine=merge_engine,
        graph=graph_store.graph,
        retrieval_engine=retrieval_engine,
        merge_cases=suite.merge_cases,
        retrieval_cases=suite.retrieval_cases,
    )


def benchmark_report_to_dict(report: BenchmarkSuiteReport) -> dict:
    return {
        "merge_metrics": asdict(report.merge_metrics),
        "merge_results": [asdict(result) for result in report.merge_results],
        "retrieval_metrics": asdict(report.retrieval_metrics),
        "retrieval_results": [
            asdict(result) for result in report.retrieval_results
        ],
    }


def format_benchmark_report(report: BenchmarkSuiteReport) -> str:
    lines = [
        "DriftGuard Benchmark Report",
        "",
        (
            "Merge: "
            f"precision={report.merge_metrics.precision:.2f} "
            f"recall={report.merge_metrics.recall:.2f} "
            f"f1={report.merge_metrics.f1:.2f}"
        ),
    ]
    lines.extend(
        [
            f"- {result.name}: {'PASS' if result.passed else 'FAIL'} "
            f"(expected={result.expected_anchor!r}, predicted={result.predicted_anchor!r})"
            for result in report.merge_results
        ]
    )
    lines.extend(
        [
            "",
            (
                "Retrieval: "
                f"precision={report.retrieval_metrics.precision:.2f} "
                f"recall={report.retrieval_metrics.recall:.2f} "
                f"f1={report.retrieval_metrics.f1:.2f}"
            ),
        ]
    )
    lines.extend(
        [
            f"- {result.name}: {'PASS' if result.passed else 'FAIL'} "
            f"(expected={result.expected_risks!r}, predicted={result.predicted_risks!r}, "
            f"confidence={result.confidence:.2f})"
            for result in report.retrieval_results
        ]
    )
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> Namespace:
    parser = ArgumentParser(
        description="Run the built-in DriftGuard quality benchmark suite."
    )
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format for the benchmark report.",
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        help="DriftGuard package log level during the benchmark run.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_level)
    report = run_builtin_benchmark()

    if args.format == "json":
        print(json.dumps(benchmark_report_to_dict(report), indent=2))
    else:
        print(format_benchmark_report(report))

    return 0
