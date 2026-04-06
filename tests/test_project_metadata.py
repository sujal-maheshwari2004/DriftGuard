from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[1]


def test_pyproject_has_publishable_metadata():
    """pyproject.toml should contain non-placeholder package metadata."""

    with open(ROOT / "pyproject.toml", "rb") as handle:
        data = tomllib.load(handle)

    project = data["project"]

    assert project["name"] == "driftguard"
    assert "Add your description here" not in project["description"]
    assert "pytest>=9.0.2" not in project["dependencies"]
    assert project["scripts"]["driftguard-mcp"] == "driftguard.server:main"
    assert "test" in project["optional-dependencies"]
    assert "demo" in project["optional-dependencies"]
    assert any(
        dependency.startswith("langgraph")
        for dependency in project["optional-dependencies"]["demo"]
    )
    assert "Homepage" in project["urls"]
    assert "Repository" in project["urls"]


def test_readme_documents_both_entrypoints():
    """README should document both the MCP and in-process guard entrypoints."""

    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "driftguard-mcp" in readme
    assert "guard_step" in readme
    assert "DriftGuardSettings" in readme
    assert "MCP Server" in readme
    assert "In-Process Guard API" in readme
    assert "End-to-End Examples" in readme
    assert "guard.record(" in readme
    assert "Local Demo" in readme
    assert "demo/rule_based" in readme
    assert "demo/langgraph" in readme
    assert ".[demo]" in readme


def test_ci_workflow_exists_and_runs_pytest_collection():
    """CI should exist and perform at least a pytest collection step."""

    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    assert "actions/setup-python" in workflow
    assert 'python-version: "3.13"' in workflow
    assert 'pip install -e ".[test]"' in workflow
    assert "python -m pytest --collect-only" in workflow
    assert "python -m pytest" in workflow


def test_gitignore_covers_core_local_artifacts():
    """.gitignore should cover the main local artifacts produced by this project."""

    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")

    assert ".pytest_cache/" in gitignore
    assert ".venv/" in gitignore
    assert "__pycache__/" in gitignore
    assert "*.egg-info/" in gitignore
    assert "driftguard_graph.json" in gitignore
    assert ".vscode/" in gitignore


def test_demo_assets_exist_and_are_documented():
    """The local demo should exist with runnable instructions."""

    demo_index = (ROOT / "demo" / "README.md").read_text(encoding="utf-8")
    rule_based_readme = (
        ROOT / "demo" / "rule_based" / "README.md"
    ).read_text(encoding="utf-8")
    langgraph_readme = (
        ROOT / "demo" / "langgraph" / "README.md"
    ).read_text(encoding="utf-8")
    demo_agent = ROOT / "demo" / "demo_agent.py"
    rule_based_agent = ROOT / "demo" / "rule_based" / "demo_agent.py"
    langgraph_agent = ROOT / "demo" / "langgraph" / "demo_agent.py"

    assert demo_agent.exists()
    assert rule_based_agent.exists()
    assert langgraph_agent.exists()
    assert "rule_based" in demo_index
    assert "langgraph" in demo_index
    assert "busy kitchen line" in rule_based_readme
    assert "demo\\rule_based\\demo_agent.py" in rule_based_readme
    assert "kitchen_line_trace.jsonl" in rule_based_readme
    assert "--runtime-mode demo" in rule_based_readme
    assert "LangGraph" in langgraph_readme
    assert "OPENAI_API_KEY" in langgraph_readme
    assert ".[demo]" in langgraph_readme
