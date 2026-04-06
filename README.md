# DriftGuard

A semantic memory system that lets AI agents **learn from mistakes and avoid repeating them**.

DriftGuard connects to any MCP-compatible agent (like Claude) and gives it two abilities: register a causal mistake, and query memory before acting to surface relevant warnings.

---

## How It Works

Every mistake is stored as a causal chain:

```
action → feedback → outcome
```

For example:

| Field    | Value           |
|----------|-----------------|
| action   | increase salt   |
| feedback | too salty       |
| outcome  | dish ruined     |

When the agent is about to do something, it queries DriftGuard with the current context. DriftGuard searches its memory graph for semantically similar past actions and returns warnings — even if the wording is different.

> *"add more salt"* will match *"increase salt"* because DriftGuard uses sentence embeddings, not keyword matching.

---

## Architecture

```
server.py               ← MCP server (FastMCP)
│
├── models/
│   ├── event.py        ← Event dataclass (action, feedback, outcome)
│   └── response.py     ← RetrievalResponse + Warning dataclasses
│
├── graph/
│   ├── graph_store.py  ← NetworkX DiGraph — core memory store
│   ├── merge_engine.py ← Semantic deduplication via embeddings
│   └── prune_engine.py ← Graph hygiene (stale/weak/isolated removal)
│
├── embedding/
│   └── embedding_engine.py  ← sentence-transformers wrapper
│
├── retrieval/
│   └── retrieval_engine.py  ← Query → warnings pipeline
│
├── storage/
│   └── persistence.py  ← JSON-based graph persistence (no pickle)
│
└── utils/
    ├── normalization.py ← spaCy lemmatization (lazy-loaded)
    └── similarity.py    ← Cosine similarity with zero-vector guard
```

---

## MCP Tools

| Tool | Description |
|---|---|
| `register_mistake` | Store a causal event (action, feedback, outcome) |
| `query_memory` | Retrieve warnings relevant to a current context |
| `deep_prune` | Run a full graph cleanup pass |
| `graph_stats` | Return current node and edge counts |

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Run the MCP server

```bash
python server.py
```

### 3. Connect to Claude Desktop

Add this to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "driftguard": {
      "command": "python",
      "args": ["/path/to/driftguard/server.py"]
    }
  }
}
```

---

## Configuration

Edit `config.py` to tune how aggressively nodes are deduplicated:

```python
SIM_THRESHOLD_ACTION   = 0.72   # lower = more merging
SIM_THRESHOLD_FEEDBACK = 0.70
SIM_THRESHOLD_OUTCOME  = 0.88   # higher = stricter (outcomes matter more)
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Design Decisions

**Why JSON instead of pickle?**
Pickle is unsafe with untrusted data, breaks across Python versions, and silently corrupts if classes are renamed. DriftGuard uses `networkx`'s node-link JSON format — human-readable, portable, and safe.

**Why sentence embeddings instead of keyword matching?**
Agents rarely repeat mistakes in identical words. Embedding similarity catches paraphrases, synonyms, and intent-equivalent actions that keyword matching would miss.

**Why a graph instead of a vector store?**
Causal chains matter. A flat vector store can tell you *"this action is risky"* but not *"this action caused this feedback which led to this outcome."* The graph preserves causality so warnings include the full chain of consequences.

**Why two prune modes?**
`light_prune` runs after every insert and stays intentionally cheap — a placeholder for hard caps or quick guards. `deep_prune` is the full cleanup and should run on a schedule. Mixing them would slow every insertion.

---

## Requirements

- Python 3.13+
- See `requirements.txt` for full dependency list