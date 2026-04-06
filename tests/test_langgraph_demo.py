from demo.langgraph.demo_agent import (
    LangGraphDemoDependencyError,
    assess_kitchen_action,
    build_langgraph_settings,
    family_name_for_action,
    parse_llm_json_response,
    should_revise_action,
    _load_langgraph_dependencies,
)


def test_langgraph_settings_reuse_demo_friendly_driftguard_config():
    """LangGraph demo settings should stay aligned with the visible demo defaults."""

    settings = build_langgraph_settings("demo/output/langgraph/test_graph.json")

    assert settings.graph_filepath == "demo/output/langgraph/test_graph.json"
    assert settings.retrieval_top_k == 4
    assert settings.retrieval_min_similarity == 0.58


def test_parse_llm_json_response_prefers_structured_action():
    """Planner parsing should recover structured JSON even when wrapped in prose."""

    parsed = parse_llm_json_response(
        'Sure, here you go:\n{"thought":"warning looks serious","action":"taste before adding more salt"}'
    )

    assert parsed["thought"] == "warning looks serious"
    assert parsed["action"] == "taste before adding more salt"


def test_langgraph_environment_marks_safe_and_risky_actions():
    """The kitchen simulator should distinguish risky actions from safe revisions."""

    risky = assess_kitchen_action("increase salt", 1)
    safe = assess_kitchen_action("taste before adding more salt", 2)

    assert risky.recorded_mistake is True
    assert risky.family_name == "seasoning"
    assert safe.recorded_mistake is False
    assert safe.family_name == "seasoning"


def test_langgraph_review_policy_only_revises_high_confidence_warnings():
    """Revision should be reserved for meaningful warning signals."""

    assert should_revise_action(warnings_count=1, review_confidence=0.80) is True
    assert should_revise_action(warnings_count=1, review_confidence=0.50) is False
    assert should_revise_action(warnings_count=0, review_confidence=0.95) is False
    assert family_name_for_action("lower heat and extend cook time") == "heat-control"


def test_langgraph_dependency_loader_fails_with_friendly_message():
    """Missing LangGraph stack should raise an actionable demo-specific error."""

    try:
        _load_langgraph_dependencies()
    except LangGraphDemoDependencyError as exc:
        message = str(exc)
    else:
        message = ""

    assert ".[demo]" in message or message == ""
