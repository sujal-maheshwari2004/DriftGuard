# Rule-Based Demo

This folder contains the original local DriftGuard demo agent built for watching the graph evolve in real time.

## Scenario

The demo simulates a busy kitchen line during dinner rush. That task is useful because it naturally creates:
- repeated paraphrased mistakes, which show semantic merge behavior
- recurring warnings, which show the guard reviewing each step
- low-frequency one-off mistakes, which get removed by scheduled pruning

## Run It

From the repository root:

```powershell
.\.venv\Scripts\python.exe .\demo\rule_based\demo_agent.py --duration-seconds 600 --step-delay 5 --prune-every 6 --reset-graph --log-level WARNING
```

The default `--runtime-mode demo` is offline-friendly and uses a small built-in semantic embedder so you can watch DriftGuard work locally without downloading the sentence-transformers model first.

If you want faster graph growth for a shorter smoke test:

```powershell
.\.venv\Scripts\python.exe .\demo\rule_based\demo_agent.py --duration-seconds 90 --step-delay 2 --prune-every 4 --reset-graph
```

If you want to compare against the full production-style stack later, run:

```powershell
.\.venv\Scripts\python.exe .\demo\rule_based\demo_agent.py --runtime-mode real --duration-seconds 90 --step-delay 2 --prune-every 4 --reset-graph
```

## What You Will See

Each step prints:
- the intended action
- the guard review result and top warning
- whether the agent kept the risky action or switched to a safer one
- graph growth after the step
- estimated merge or edge reuse
- scheduled prune activity
- final graph stats for that step

The demo also writes:
- `demo/output/rule_based/kitchen_line_graph.json`: the persisted graph
- `demo/output/rule_based/kitchen_line_trace.jsonl`: one JSON record per step

## Notes

- Noise-injection steps intentionally create weak one-off memories so pruning has something visible to remove.
- The demo uses a separate graph file from the package default so it does not pollute your main local memory store.
