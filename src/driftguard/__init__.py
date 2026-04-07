__version__ = "0.1.0"

from driftguard.config import DEFAULT_SETTINGS, DriftGuardSettings
from driftguard.benchmark import (
    BenchmarkMergeEngine,
    BenchmarkSuite,
    BenchmarkEmbeddingEngine,
    benchmark_report_to_dict,
    builtin_benchmark_suite,
    build_benchmark_runtime,
    format_benchmark_report,
    run_builtin_benchmark,
)
from driftguard.adapters.generic import review_payload
from driftguard.adapters.langgraph import make_langgraph_review_node
from driftguard.errors import (
    DriftGuardDependencyError,
    DriftGuardError,
    EmbeddingDependencyError,
    NormalizationDependencyError,
)
from driftguard.evaluation import (
    BenchmarkSuiteReport,
    MergeBenchmarkCase,
    MergeBenchmarkResult,
    MetricSummary,
    RetrievalBenchmarkCase,
    RetrievalBenchmarkResult,
    evaluate_benchmark_suite,
    evaluate_merge_cases,
    evaluate_retrieval_cases,
    metric_summary,
)
from driftguard.logging_config import configure_logging
from driftguard.metrics import DriftGuardMetrics, MetricsSnapshot
from driftguard.guard import (
    DriftGuard,
    GuardrailAcknowledgementRequired,
    GuardrailTriggered,
    guard_step,
)
from driftguard.mcp import create_mcp_server
from driftguard.runtime import DriftGuardRuntime, build_runtime
from driftguard.storage.sqlite_persistence import SQLitePersistence
