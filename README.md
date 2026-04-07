# DriftGuard

DriftGuard is a semantic mistake-memory and guardrail layer for AI agents.

It sits between agent intent and agent execution:
- review the next planned step
- surface warnings from similar past failures
- optionally block or require acknowledgement
- record new mistakes after the step completes

DriftGuard is framework-agnostic at the integration level. Today it supports two primary entrypoints:
- `driftguard-mcp` for MCP-capable agents
- `DriftGuard` / `guard_step(...)` for in-process Python agents

## Why Use It

DriftGuard is useful when an agent can already act, but you want it to stop repeating the same mistakes.

It stores causal memories in this shape:

```text
action -> feedback -> outcome
```

Example:
- action: `increase salt`
- feedback: `too salty`
- outcome: `dish ruined`

Because DriftGuard uses semantic matching instead of exact string matching, `"add more salt"` can still retrieve warnings learned from `"increase salt"`.

## What DriftGuard Can Do

- semantically merge similar mistake memories
- retrieve warnings before an agent step executes
- support `warn`, `block`, `acknowledge`, and `record_only` guard policies
- persist memory to JSON or SQLite while keeping the same graph-based runtime model
- prune weak, stale, and isolated graph structure
- expose runtime metrics and graph stats
- benchmark merge and retrieval quality with a built-in offline suite
- integrate through MCP, plain Python, or helper adapters for generic payloads and LangGraph-style state

## Installation

Install the package:

```bash
pip install .
```

Install test dependencies:

```bash
pip install ".[test]"
```

Install demo dependencies for the LangGraph walkthrough:

```bash
pip install ".[demo]"
```

Install the spaCy model used for normalization:

```bash
python -m spacy download en_core_web_sm
```

If DriftGuard cannot load the embedding backend or spaCy model, it raises DriftGuard-specific dependency errors with setup guidance.

## Quick Start

### Option 1: MCP Server

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
- `guard_metrics`

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

### Option 2: In-Process Guard API

Use DriftGuard directly inside a Python agent loop:

```python
from driftguard import DriftGuard

guard = DriftGuard()

review = guard.before_step("increase salt")
if review.warnings:
    print(review.warnings[0].risk)

guard.record(
    action="increase salt",
    feedback="too salty",
    outcome="dish ruined",
)
```

Decorator form:

```python
from driftguard import DriftGuard, guard_step

guard = DriftGuard()

@guard_step(guard, input_getter=lambda payload: payload["next_action"])
def agent_step(payload: dict):
    return payload["next_action"]
```

## Guard Policies

The in-process guard supports four policy modes:
- `warn`: review and continue
- `block`: raise if confidence crosses the threshold
- `acknowledge`: require explicit acknowledgement before continuing
- `record_only`: skip pre-step review and only use DriftGuard for recording

Example:

```python
from driftguard import DriftGuard, DriftGuardSettings

guard = DriftGuard(
    settings=DriftGuardSettings(
        guard_policy="acknowledge",
        guard_min_confidence=0.8,
    )
)
```

## Adapters

If you do not want to wire everything manually, DriftGuard includes small adapter helpers.

Generic payload review:

```python
from driftguard import DriftGuard, review_payload

guard = DriftGuard()
result = review_payload(
    guard,
    {"action": "increase salt", "attempt": 2},
)
```

LangGraph-style state review node:

```python
from driftguard import DriftGuard, make_langgraph_review_node

guard = DriftGuard()
review_node = make_langgraph_review_node(guard)

state_update = review_node({"candidate_action": "increase salt"})
```

## Configuration

DriftGuard uses a shared settings object:

```python
from driftguard import DriftGuardSettings, build_runtime

settings = DriftGuardSettings(
    graph_filepath="driftguard_graph.json",
    storage_backend="sqlite",
    sqlite_filepath="driftguard_graph.sqlite3",
    retrieval_top_k=5,
    retrieval_min_similarity=0.60,
    retrieval_recency_weight=0.15,
    similarity_threshold_action=0.72,
    guard_policy="warn",
)

runtime = build_runtime(settings=settings)
```

Useful settings include:
- `graph_filepath`
- `storage_backend`
- `sqlite_filepath`
- `embedding_model_name`
- `embedding_device`
- `retrieval_top_k`
- `retrieval_min_similarity`
- `retrieval_recency_weight`
- `traversal_max_depth`
- `traversal_max_branching`
- `traversal_max_paths`
- `similarity_threshold_action`
- `similarity_threshold_feedback`
- `similarity_threshold_outcome`
- `guard_policy`
- `guard_min_confidence`
- `prune_node_stale_days`
- `prune_edge_min_frequency`
- `log_level`

Use `storage_backend="sqlite"` to keep the same graph model while persisting it to SQLite instead of JSON.

## Storage Model

DriftGuard uses a graph in memory and a persistence backend underneath it.

- in-memory model: graph-based runtime using `networkx`
- persistence backends: versioned JSON or SQLite

This means you keep graph behavior while choosing a storage backend that fits local or more serious usage.

## Metrics and Observability

DriftGuard now tracks lightweight runtime metrics such as:
- review counts
- warnings surfaced
- blocked steps
- acknowledgement-required steps
- skipped reviews
- stored records
- node and edge creation or reuse
- prune activity

In Python:

```python
from driftguard import build_runtime

runtime = build_runtime()
snapshot = runtime.metrics_snapshot()
print(snapshot["counters"])
```

Over MCP:
- call `guard_metrics`

## Evaluation and Benchmarking

DriftGuard includes a lightweight evaluation harness plus a built-in offline benchmark suite.

You can benchmark:
- merge precision / recall / F1
- retrieval precision / recall / F1

Run the built-in benchmark:

```bash
driftguard-benchmark
```

Emit structured JSON:

```bash
driftguard-benchmark --format json
```

You can also define your own merge and retrieval benchmark cases through the Python evaluation API.

## Local Demos

There are two demo tracks under [demo/README.md](demo/README.md):
- `demo/rule_based/`: deterministic simulator for graph growth, merge behavior, and pruning
- `demo/langgraph/`: LangGraph-based LLM agent that DriftGuard reviews before each step

Rule-based walkthrough:

```powershell
.\.venv\Scripts\python.exe .\demo\rule_based\demo_agent.py --duration-seconds 600 --step-delay 5 --prune-every 6 --reset-graph --log-level WARNING
```

LangGraph LLM walkthrough:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[demo]"
.\.venv\Scripts\python.exe .\demo\langgraph\demo_agent.py --duration-seconds 120 --step-delay 4 --prune-every 4 --reset-graph
```

The rule-based demo defaults to an offline-friendly built-in runtime so you can exercise DriftGuard locally even without the Hugging Face model cached. The LangGraph demo uses an OpenAI-compatible chat model and reads `OPENAI_API_KEY` plus optional `OPENAI_MODEL` / `OPENAI_BASE_URL`.

## Architecture

Current package layout:

```text
src/driftguard/
  adapters/
  benchmark.py
  config.py
  errors.py
  evaluation.py
  guard.py
  logging_config.py
  mcp.py
  metrics.py
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
- `metrics.py`: counters and gauges for runtime behavior
- `evaluation.py`: merge/retrieval evaluation primitives
- `benchmark.py`: built-in benchmark suite and CLI
- `storage/persistence.py`: versioned JSON persistence
- `storage/sqlite_persistence.py`: SQLite persistence backend
- `graph/merge_engine.py`: semantic matching and deduplication
- `retrieval/retrieval_engine.py`: warning generation and confidence scoring

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

DriftGuard is an early but already usable release candidate for a focused agent-memory niche. The project currently includes:
- real `src` package layout
- dual entrypoints
- shared runtime architecture
- JSON and SQLite persistence
- centralized logging
- policy-driven in-process guardrails
- metrics and graph stats
- evaluation and benchmark tooling
- demos for rule-based and LangGraph usage
- targeted pytest coverage across core behaviors

## CI

The repository includes a GitHub Actions workflow that:
- installs the package with test dependencies
- collects the test suite
- runs the full pytest suite
