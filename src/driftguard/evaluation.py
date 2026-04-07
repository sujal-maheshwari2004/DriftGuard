from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MetricSummary:
    true_positive: int
    false_positive: int
    false_negative: int
    precision: float
    recall: float
    f1: float


@dataclass(frozen=True)
class MergeBenchmarkCase:
    name: str
    query: str
    node_type: str
    expected_anchor: str | None


@dataclass(frozen=True)
class MergeBenchmarkResult:
    name: str
    query: str
    expected_anchor: str | None
    predicted_anchor: str | None
    passed: bool


@dataclass(frozen=True)
class RetrievalBenchmarkCase:
    name: str
    query: str
    expected_risks: tuple[str, ...]


@dataclass(frozen=True)
class RetrievalBenchmarkResult:
    name: str
    query: str
    expected_risks: tuple[str, ...]
    predicted_risks: tuple[str, ...]
    confidence: float
    passed: bool


@dataclass(frozen=True)
class BenchmarkSuiteReport:
    merge_metrics: MetricSummary
    merge_results: tuple[MergeBenchmarkResult, ...]
    retrieval_metrics: MetricSummary
    retrieval_results: tuple[RetrievalBenchmarkResult, ...]


def metric_summary(
    true_positive: int,
    false_positive: int,
    false_negative: int,
) -> MetricSummary:
    precision = _safe_divide(true_positive, true_positive + false_positive)
    recall = _safe_divide(true_positive, true_positive + false_negative)
    f1 = _safe_divide(2 * precision * recall, precision + recall)
    return MetricSummary(
        true_positive=true_positive,
        false_positive=false_positive,
        false_negative=false_negative,
        precision=precision,
        recall=recall,
        f1=f1,
    )


def evaluate_merge_cases(
    merge_engine,
    graph,
    cases: list[MergeBenchmarkCase] | tuple[MergeBenchmarkCase, ...],
) -> tuple[MetricSummary, tuple[MergeBenchmarkResult, ...]]:
    true_positive = 0
    false_positive = 0
    false_negative = 0
    results: list[MergeBenchmarkResult] = []

    for case in cases:
        normalized_query = merge_engine.normalize(case.query)
        predicted_anchor = merge_engine.find_similar_node(
            normalized_query,
            case.node_type,
            graph,
        )

        if case.expected_anchor is None:
            passed = predicted_anchor is None
            if predicted_anchor is not None:
                false_positive += 1
        else:
            passed = predicted_anchor == case.expected_anchor
            if passed:
                true_positive += 1
            else:
                false_negative += 1
                if predicted_anchor is not None:
                    false_positive += 1

        results.append(
            MergeBenchmarkResult(
                name=case.name,
                query=case.query,
                expected_anchor=case.expected_anchor,
                predicted_anchor=predicted_anchor,
                passed=passed,
            )
        )

    return metric_summary(true_positive, false_positive, false_negative), tuple(
        results
    )


def evaluate_retrieval_cases(
    retrieval_engine,
    cases: list[RetrievalBenchmarkCase] | tuple[RetrievalBenchmarkCase, ...],
) -> tuple[MetricSummary, tuple[RetrievalBenchmarkResult, ...]]:
    true_positive = 0
    false_positive = 0
    false_negative = 0
    results: list[RetrievalBenchmarkResult] = []

    for case in cases:
        response = retrieval_engine.query(case.query)
        predicted_risks = tuple(warning.risk for warning in response.warnings)
        predicted_risk_set = set(predicted_risks)
        expected_risk_set = set(case.expected_risks)

        true_positive += len(predicted_risk_set & expected_risk_set)
        false_positive += len(predicted_risk_set - expected_risk_set)
        false_negative += len(expected_risk_set - predicted_risk_set)

        results.append(
            RetrievalBenchmarkResult(
                name=case.name,
                query=case.query,
                expected_risks=case.expected_risks,
                predicted_risks=predicted_risks,
                confidence=response.confidence,
                passed=predicted_risk_set == expected_risk_set,
            )
        )

    return metric_summary(true_positive, false_positive, false_negative), tuple(
        results
    )


def evaluate_benchmark_suite(
    *,
    merge_engine,
    graph,
    retrieval_engine,
    merge_cases: list[MergeBenchmarkCase] | tuple[MergeBenchmarkCase, ...],
    retrieval_cases: list[RetrievalBenchmarkCase]
    | tuple[RetrievalBenchmarkCase, ...],
) -> BenchmarkSuiteReport:
    merge_metrics, merge_results = evaluate_merge_cases(
        merge_engine,
        graph,
        merge_cases,
    )
    retrieval_metrics, retrieval_results = evaluate_retrieval_cases(
        retrieval_engine,
        retrieval_cases,
    )
    return BenchmarkSuiteReport(
        merge_metrics=merge_metrics,
        merge_results=merge_results,
        retrieval_metrics=retrieval_metrics,
        retrieval_results=retrieval_results,
    )


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator
