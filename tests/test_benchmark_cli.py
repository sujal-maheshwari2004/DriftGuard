import json

from driftguard.benchmark import (
    benchmark_report_to_dict,
    builtin_benchmark_suite,
    format_benchmark_report,
    main,
    parse_args,
    run_builtin_benchmark,
)


def test_builtin_benchmark_suite_contains_seed_events_and_cases():
    """The built-in benchmark should ship with reusable merge and retrieval cases."""

    suite = builtin_benchmark_suite()

    assert len(suite.seed_events) >= 4
    assert len(suite.merge_cases) >= 3
    assert len(suite.retrieval_cases) >= 4


def test_run_builtin_benchmark_produces_nontrivial_scores():
    """The built-in benchmark should be runnable offline and yield meaningful metrics."""

    report = run_builtin_benchmark()

    assert report.merge_metrics.precision >= 0.5
    assert report.retrieval_metrics.recall >= 0.5
    assert any(result.passed for result in report.merge_results)
    assert any(result.passed for result in report.retrieval_results)


def test_format_benchmark_report_contains_summary_sections():
    """Text reports should include both merge and retrieval summary sections."""

    report = run_builtin_benchmark()
    rendered = format_benchmark_report(report)

    assert "DriftGuard Benchmark Report" in rendered
    assert "Merge:" in rendered
    assert "Retrieval:" in rendered


def test_benchmark_report_to_dict_is_json_serializable():
    """Structured reports should serialize cleanly for machine-readable output."""

    payload = benchmark_report_to_dict(run_builtin_benchmark())

    assert "merge_metrics" in payload
    assert "retrieval_metrics" in payload
    json.dumps(payload)


def test_benchmark_cli_supports_json_output(capsys):
    """The benchmark CLI should emit machine-readable JSON when requested."""

    exit_code = main(["--format", "json"])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert "merge_metrics" in payload
    assert "retrieval_metrics" in payload


def test_benchmark_cli_parser_defaults_to_text():
    """The benchmark CLI should default to human-readable text output."""

    args = parse_args([])

    assert args.format == "text"
