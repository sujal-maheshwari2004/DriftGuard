# LangGraph Demo

This demo runs a real LLM-driven agent loop on top of LangGraph while DriftGuard reviews each proposed step before execution.

## What It Does

- uses a LangGraph state graph to plan, review, revise, and execute actions
- uses an OpenAI-compatible chat model for planning
- runs DriftGuard before each proposed action
- records risky outcomes back into the graph
- prunes on a schedule so you can see the memory stay tidy during the run

## Install Demo Dependencies

From the repository root:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[demo]"
```

Then set your model credentials:

```powershell
$env:OPENAI_API_KEY="your-key"
$env:OPENAI_MODEL="gpt-4o-mini"
```

Optional:

```powershell
$env:OPENAI_BASE_URL="https://your-openai-compatible-endpoint/v1"
```

## Run It

The 120-second walkthrough:

```powershell
.\.venv\Scripts\python.exe .\demo\langgraph\demo_agent.py --duration-seconds 120 --step-delay 4 --prune-every 4 --reset-graph
```

This writes:
- `demo/output/langgraph/kitchen_line_graph.json`
- `demo/output/langgraph/kitchen_line_trace.jsonl`

## Notes

- `--runtime-mode demo` keeps DriftGuard itself offline-friendly while the LLM still comes from your configured OpenAI-compatible endpoint.
- `--runtime-mode real` switches DriftGuard back to the package's full embedding and normalization stack.
- If you only want the old deterministic simulator, use the rule-based demo instead.
