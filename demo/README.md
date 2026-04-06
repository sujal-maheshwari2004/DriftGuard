# Demo Index

DriftGuard now ships with two local demo tracks:

- [rule_based/README.md](rule_based/README.md): the original deterministic simulator that is great for watching graph growth, merge behavior, and pruning quickly
- [langgraph/README.md](langgraph/README.md): an LLM-driven LangGraph agent that reviews each step with DriftGuard before acting

## Quick Start

Rule-based walkthrough:

```powershell
.\.venv\Scripts\python.exe .\demo\rule_based\demo_agent.py --duration-seconds 120 --step-delay 2 --prune-every 4 --reset-graph
```

LangGraph walkthrough:

```powershell
.\.venv\Scripts\python.exe .\demo\langgraph\demo_agent.py --duration-seconds 120 --step-delay 4 --prune-every 4 --reset-graph
```

The old `.\demo\demo_agent.py` path still works as a compatibility wrapper for the rule-based demo.
