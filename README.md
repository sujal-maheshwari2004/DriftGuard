# DriftGuard

**DriftGuard** is a semantic mistake-memory and guardrail layer for autonomous agents.

It sits between **intent** and **execution**, allowing agents to learn from past failures and avoid repeating them.

DriftGuard stores structured causal memories:

```
action → feedback → outcome
```

and surfaces warnings when similar risky actions appear again.

It works with:

* MCP agents
* LangGraph workflows
* custom Python agents
* tool-calling planners
* autonomous pipelines

---

## Why DriftGuard Exists

Agents today can act.

They usually cannot **remember mistakes meaningfully**.

Typical failure loop:

```
agent makes mistake
agent retries
agent repeats mistake
agent retries again
```

DriftGuard introduces a semantic failure memory layer:

```
plan step
↓
DriftGuard review
↓
warning surfaced
↓
agent revises action
```

This improves:

* stability
* reliability
* convergence speed
* evaluation consistency
* production safety

without requiring changes to your planner architecture.

---

## What DriftGuard Does

DriftGuard provides:

• semantic mistake memory
• similarity-aware warning retrieval
• policy-based execution guardrails
• merge + deduplicate memory graphs
• JSON or SQLite persistence
• runtime metrics
• pruning of stale weak memories
• MCP server integration
• LangGraph adapters
• offline benchmark harness

---

## Installation

Install from PyPI:

```bash
pip install driftguard
```

Install test dependencies:

```bash
pip install "driftguard[test]"
```

Install LangGraph demo dependencies:

```bash
pip install "driftguard[demo]"
```

Install the spaCy normalization model:

```bash
python -m spacy download en_core_web_sm
```

---

## Quick Example (Python Agent)

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

DriftGuard now remembers this failure and warns on similar steps later.

---

## Guard Policies

Control how the agent reacts to detected risks:

```python
from driftguard import DriftGuard, DriftGuardSettings

guard = DriftGuard(
    settings=DriftGuardSettings(
        guard_policy="acknowledge",
        guard_min_confidence=0.8,
    )
)
```

Supported modes:

| policy      | behavior                     |
| ----------- | ---------------------------- |
| warn        | surface warning only         |
| block       | raise exception              |
| acknowledge | require confirmation         |
| record_only | store memory but skip review |

---

## MCP Server Usage

Run DriftGuard as an MCP server:

```bash
driftguard-mcp
```

Available tools:

```
register_mistake
query_memory
deep_prune
graph_stats
guard_metrics
```

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

---

## LangGraph Integration

Create a review node inside a LangGraph workflow:

```python
from driftguard import DriftGuard
from driftguard import make_langgraph_review_node

guard = DriftGuard()

review_node = make_langgraph_review_node(guard)
```

Drop this node directly into a planner graph.

---

## Generic Payload Adapter

Review arbitrary planner payloads:

```python
from driftguard import DriftGuard, review_payload

guard = DriftGuard()

result = review_payload(
    guard,
    {"action": "increase salt", "attempt": 2},
)
```

---

## CLI Benchmark Tool

Evaluate merge and retrieval quality:

```bash
driftguard-benchmark
```

Export structured results:

```bash
driftguard-benchmark --format json
```

Measures:

* merge precision
* merge recall
* retrieval precision
* retrieval recall
* F1 score

---

## Storage Model

DriftGuard uses:

```
in-memory semantic graph runtime
+
persistent storage backend
```

Supported persistence:

| backend | purpose              |
| ------- | -------------------- |
| JSON    | local experiments    |
| SQLite  | production workflows |

Example configuration:

```python
from driftguard import DriftGuardSettings

settings = DriftGuardSettings(
    storage_backend="sqlite",
    sqlite_filepath="driftguard.sqlite3",
)
```

---

## Metrics and Observability

Runtime metrics available:

```python
from driftguard import build_runtime

runtime = build_runtime()

snapshot = runtime.metrics_snapshot()

print(snapshot["counters"])
```

Includes:

```
reviews
warnings
blocks
acknowledgements
records
node reuse
edge reuse
prune activity
```

Also available via MCP:

```
guard_metrics
```

---

## Example Architecture Placement

Typical agent loop:

```
planner
 ↓
candidate action
 ↓
DriftGuard review
 ↓
warning surfaced
 ↓
planner revision
 ↓
execution
 ↓
feedback recorded
```

DriftGuard improves stability without replacing the planner.

---

## Local Demos

Two included demos:

### Rule-based simulator

Offline deterministic walkthrough:

```bash
python demo/rule_based/demo_agent.py
```

Shows:

* merge behavior
* warning retrieval
* pruning cleanup
* graph evolution

---

### LangGraph LLM agent demo

```bash
pip install "driftguard[demo]"
python demo/langgraph/demo_agent.py
```

Demonstrates:

planner → guard → revise → execute loop

with real model interaction.

---

## CLI Entry Points

Installed automatically:

```
driftguard-mcp
driftguard-benchmark
```

---

## Configuration Surface

Example advanced setup:

```python
from driftguard import DriftGuardSettings

settings = DriftGuardSettings(
    retrieval_top_k=5,
    retrieval_min_similarity=0.60,
    similarity_threshold_action=0.72,
    guard_policy="warn",
)
```

Full configuration supports:

```
retrieval tuning
similarity thresholds
guard policy modes
storage backend selection
embedding configuration
graph pruning controls
logging verbosity
```

---

## When To Use DriftGuard

DriftGuard helps when your agent:

* retries failing steps repeatedly
* forgets past execution errors
* needs execution-time guardrails
* requires semantic mistake recall
* runs multi-step planners
* uses LangGraph or MCP
* executes tools autonomously

---

## Project Status

Current release includes:

* semantic merge engine
* similarity retrieval engine
* graph persistence layer
* SQLite backend
* MCP server
* LangGraph adapter
* benchmark harness
* runtime metrics
* pruning engine
* deterministic demo runtime
* pytest coverage

DriftGuard is suitable for early production experimentation and agent-infrastructure research workflows.

---

## License

MIT License