from pathlib import Path

from demo.rule_based.demo_agent import (
    DemoEmbeddingEngine,
    build_demo_guard,
    build_demo_settings,
    build_step_plan,
    phase_for_step,
    should_prune,
    should_switch_to_safe_action,
    simple_normalize_text,
    summarize_event_growth,
)
from driftguard.utils.similarity import cosine_similarity


ROOT = Path(__file__).resolve().parents[1]


def test_build_demo_settings_use_demo_friendly_defaults():
    """Demo settings should tune retrieval and pruning for visible local behavior."""

    settings = build_demo_settings("demo/output/test_graph.json", log_level="INFO")

    assert settings.graph_filepath == "demo/output/test_graph.json"
    assert settings.retrieval_top_k == 4
    assert settings.retrieval_min_similarity == 0.58
    assert settings.prune_edge_min_frequency == 2
    assert settings.traversal_max_depth == 2
    assert settings.log_level == "INFO"


def test_phase_and_plan_switch_to_noise_every_seventh_step():
    """Every seventh step should inject a one-off noisy mistake for prune visibility."""

    assert phase_for_step(7) == "noise-injection"

    plan = build_step_plan(7)

    assert plan.phase == "noise-injection"
    assert plan.family_name == "noise"
    assert "ask the lead cook" in plan.safe_action


def test_switch_to_safe_action_policy_changes_by_phase():
    """The demo should ignore early warnings, then increasingly heed them later."""

    assert should_switch_to_safe_action(
        4,
        phase="seed-memory",
        has_warning=True,
    ) is False
    assert should_switch_to_safe_action(
        12,
        phase="reinforce-patterns",
        has_warning=True,
    ) is True
    assert should_switch_to_safe_action(
        13,
        phase="reinforce-patterns",
        has_warning=True,
    ) is False
    assert should_switch_to_safe_action(
        21,
        phase="guided-recovery",
        has_warning=True,
    ) is True
    assert should_switch_to_safe_action(
        25,
        phase="guided-recovery",
        has_warning=True,
    ) is False


def test_summarize_event_growth_estimates_merge_activity():
    """Growth summaries should expose likely merges and edge reuse for the demo output."""

    summary = summarize_event_growth(
        {"nodes": 10, "edges": 7},
        {"nodes": 11, "edges": 8},
        recorded=True,
    )

    assert summary == {
        "delta_nodes": 1,
        "delta_edges": 1,
        "estimated_merged_nodes": 2,
        "estimated_reused_edges": 1,
    }

    assert should_prune(6, 6) is True
    assert should_prune(5, 6) is False


def test_demo_runtime_is_offline_friendly_and_semantic():
    """The demo runtime should work without external model downloads and still match paraphrases."""

    graph_path = ROOT / "demo" / "output" / "pytest_demo_runtime_graph.json"

    if graph_path.exists():
        graph_path.unlink()

    try:
        guard = build_demo_guard(build_demo_settings(str(graph_path)))

        guard.record(
            action="increase salt",
            feedback="too salty",
            outcome="dish ruined",
        )
        review = guard.before_step("add more salt")

        assert review.warnings
        assert review.warnings[0].risk == "salt"
        assert guard.stats()["nodes"] >= 3
    finally:
        if graph_path.exists():
            graph_path.unlink()


def test_demo_normalization_preserves_shared_concepts():
    """The built-in demo normalizer should keep paraphrased domain concepts aligned."""

    normalized_a = simple_normalize_text("add more salt")
    normalized_b = simple_normalize_text("season more aggressively")
    normalized_c = simple_normalize_text("rest the protein before plating")
    embedding_engine = DemoEmbeddingEngine()

    assert "salt" in normalized_a
    assert "salt" in normalized_b
    assert "rest" in normalized_c

    similarity = cosine_similarity(
        embedding_engine.embed("add more salt"),
        embedding_engine.embed("season more aggressively"),
    )
    unrelated = cosine_similarity(
        embedding_engine.embed("add more salt"),
        embedding_engine.embed("rest the protein before plating"),
    )

    assert similarity > unrelated


def test_legacy_demo_wrapper_still_exports_rule_based_main():
    """The old demo entrypoint should remain as a compatibility wrapper."""

    from demo.demo_agent import main as legacy_main
    from demo.rule_based.demo_agent import main as canonical_main

    assert legacy_main is canonical_main
