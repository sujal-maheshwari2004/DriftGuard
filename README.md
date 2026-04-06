# DriftGuard

DriftGuard is a semantic mistake-memory system for AI agents. It helps agents remember what went wrong before and avoid repeating the same mistake again.

It supports two integration styles:
- `driftguard-mcp`: an MCP server for tool-based agent integrations
- `DriftGuard` / `guard_step(...)`: an in-process guardrail API for reviewing actions before each agent step

## What It Stores

Each memory is a causal chain:

```text
action -> feedback -> outcome
```

Example:
- action: `increase salt`
- feedback: `too salty`
- outcome: `dish ruined`

Before an agent acts, DriftGuard can query past memories and return warnings based on semantic similarity instead of exact wording. That means `"add more salt"` can still match `"increase salt"`.

## Entry Points

### 1. MCP Server

Run the packaged MCP entrypoint:

```bash
driftguard-mcp
```

Or locally from the repo:

```bash
python server.py
```

Available MCP tools:
- `register_mistake`
- `query_memory`
- `deep_prune`
- `graph_stats`

Example Claude Desktop config:

```json
{
  "mcpServers": {
    "driftguard": {
      "command": "driftguard-mcp"
    }
  }
}
```

### 2. In-Process Guard API

Use the shared runtime directly inside an agent loop:

```python
from driftguard import DriftGuard

guard = DriftGuard()

review = guard.before_step("increase salt")
if review.warnings:
    print(review.warnings[0].risk)
```

Decorator form:

```python
from driftguard import DriftGuard, guard_step

guard = DriftGuard()

@guard_step(guard, input_getter=lambda payload: payload["next_action"])
def agent_step(payload: dict):
    return payload["next_action"]
```

## Architecture

Current package layout:

```text
src/driftguard/
  config.py
  errors.py
  guard.py
  logging_config.py
  mcp.py
  runtime.py
  server.py
  embedding/
  graph/
  models/
  retrieval/
  storage/
  utils/
```

Key pieces:
- `runtime.py`: shared construction and operational core
- `mcp.py`: MCP server factory
- `guard.py`: in-process guard and decorator entrypoint
- `storage/persistence.py`: versioned JSON persistence with backward-compatible loads
- `graph/merge_engine.py`: semantic matching and deduplication
- `retrieval/retrieval_engine.py`: warning generation and confidence scoring

## Installation

Install the package:

```bash
pip install .
```

Install test dependencies:

```bash
pip install ".[test]"
```

Install the spaCy model used for normalization:

```bash
python -m spacy download en_core_web_sm
```

If DriftGuard cannot load the embedding model or spaCy model, it now raises DriftGuard-specific dependency errors with setup guidance.

## Configuration

DriftGuard uses a shared settings object:

```python
from driftguard import DriftGuardSettings, build_runtime

settings = DriftGuardSettings(
    graph_filepath="driftguard_graph.json",
    retrieval_top_k=5,
    retrieval_min_similarity=0.60,
    similarity_threshold_action=0.72,
)

runtime = build_runtime(settings=settings)
```

Useful settings include:
- `graph_filepath`
- `embedding_model_name`
- `embedding_device`
- `retrieval_top_k`
- `retrieval_min_similarity`
- `similarity_threshold_action`
- `similarity_threshold_feedback`
- `similarity_threshold_outcome`
- `prune_node_stale_days`
- `prune_edge_min_frequency`
- `log_level`

## Development

Run tests:

```bash
python -m pytest
```

Collect tests only:

```bash
python -m pytest --collect-only
```

## Current Status

DriftGuard is still early-stage, but the project now includes:
- real `src` package layout
- shared runtime architecture
- dual entrypoints
- centralized logging
- versioned persistence
- targeted pytest coverage for guardrails, retrieval precision, persistence hardening, dependency failures, and runtime/MCP wiring

## Remaining Work

The main remaining publishability work is:
- final package metadata polish
- broader CI coverage across environments
- more end-to-end usage examples
