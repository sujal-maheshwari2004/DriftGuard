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


def test_ci_workflow_exists_and_runs_pytest_collection():
    """CI should exist and perform at least a pytest collection step."""

    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    assert "actions/setup-python" in workflow
    assert 'python-version: "3.13"' in workflow
    assert 'pip install -e ".[test]"' in workflow
    assert "python -m pytest --collect-only" in workflow
