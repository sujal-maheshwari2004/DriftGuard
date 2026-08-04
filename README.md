# DriftGuard

**A semantic mistake-memory and guardrail layer for autonomous agents.**

DriftGuard sits between **intent** and **execution**. Before your agent runs a step, DriftGuard checks whether something semantically similar has failed before — and says so.

```
                    ┌─────────────────────────┐
   candidate        │                         │        warning +
   action    ─────► │       DriftGuard        │ ─────► reinforcement
                    │  "have we been here?"   │
                    └─────────────────────────┘
```

[![PyPI](https://img.shields.io/pypi/v/driftguard-ai)](https://pypi.org/project/driftguard-ai/)
[![Python](https://img.shields.io/pypi/pyversions/driftguard-ai)](https://pypi.org/project/driftguard-ai/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## Table of Contents

- [The Problem](#the-problem)
- [How It Works — In One Picture](#how-it-works--in-one-picture)
- [Install](#install)
- [Quickstart](#quickstart)
- [Worked Example: Watch The Memory Form](#worked-example-watch-the-memory-form)
- [High-Level Design (HLD)](#high-level-design-hld)
- [Low-Level Design (LLD)](#low-level-design-lld)
- [Data Model](#data-model)
- [Confidence Scoring, Explained](#confidence-scoring-explained)
- [Guard Policies](#guard-policies)
- [Integrations](#integrations)
- [Storage Backends](#storage-backends)
- [Configuration Reference](#configuration-reference)
- [Metrics Reference](#metrics-reference)
- [Benchmarks and Honest Limitations](#benchmarks-and-honest-limitations)
- [Project Status](#project-status)
- [Contributing](#contributing)

---

## The Problem

An agent that can act but cannot remember failing produces this loop:

```
attempt 1  ──►  fails  ──►  retry
attempt 2  ──►  fails  ──►  retry          ← same mistake, new wording
attempt 3  ──►  fails  ──►  retry
```

Retrying is cheap. Retrying the *same wrong idea in different words* is what burns tokens and breaks production runs. Conversation history doesn't fix this: it's linear, it gets truncated, and "delete the temp directory" does not textually match "clear out /tmp" even though they are the same catastrophic step.

DriftGuard stores failures as **causal chains** and retrieves them by **meaning**, not by string match:

```
action  ──►  feedback  ──►  outcome
```

---

## How It Works — In One Picture

```mermaid
flowchart LR
    P[Planner] -->|candidate action| G{DriftGuard<br/>review}
    G -->|no match| E[Execute]
    G -->|warning| R[Planner revises]
    R --> E
    E -->|result observed| REC[Record outcome]
    REC -.->|grows memory| G

    style G fill:#fff3cd,stroke:#856404,stroke-width:2px
    style E fill:#d4edda,stroke:#155724
    style REC fill:#e2e3ff,stroke:#383d7c
```

DriftGuard **does not replace your planner**. It is a read/write memory the planner consults. Two calls are the whole surface area:

| Call | When | Effect |
| --- | --- | --- |
| `guard.before_step(action)` | Before executing | Returns warnings + reinforcements |
| `guard.record(...)` / `guard.record_success(...)` | After observing a result | Grows the memory |

---

## Install

```bash
pip install driftguard-ai
```

DriftGuard needs a spaCy model for text normalization:

```bash
python -m spacy download en_core_web_sm
```

Optional extras:

```bash
pip install "driftguard-ai[test]"      # pytest
pip install "driftguard-ai[demo]"      # LangGraph + langchain-openai demo
pip install "driftguard-ai[postgres]"  # SQLAlchemy + psycopg
```

> **Install size note:** the base install pulls `sentence-transformers` (and therefore PyTorch) and `spacy`. Expect a multi-gigabyte environment. If that's a problem for your deployment, see [Known Limitations](#known-limitations) — making the embedding backend pluggable is on the roadmap.

The import name is `driftguard`; the distribution name is `driftguard-ai`.

---

## Quickstart

```python
from driftguard import DriftGuard

guard = DriftGuard()

# 1. Ask before acting
review = guard.before_step("retry the payment webhook with the same payload")

for warning in review.warnings:
    print(f"⚠️  {warning.trigger} → {warning.risk}  (confidence {warning.confidence:.2f})")

for reinforcement in review.reinforcements:
    print(f"✅ {reinforcement.trigger} → {reinforcement.recommendation}")

# 2. Tell it what happened
guard.record(
    action="retry the payment webhook with the same payload",
    feedback="server returned 422 again",
    outcome="duplicate charge risk, run aborted",
)
```

The next time the agent proposes anything close to that action — *"send the webhook again"*, *"re-fire the payment callback"* — DriftGuard surfaces the warning, without those strings ever matching.

---

## Worked Example: Watch The Memory Form

This is the clearest way to understand DriftGuard. Follow the graph as it changes.

### Step 1 — First failure recorded

```python
guard.record(
    action="delete the temp directory",
    feedback="build cache was wiped",
    outcome="next build took 40 minutes",
)
```

Text is normalized (lowercased, lemmatized, stopwords stripped), embedded, and stored as three linked nodes:

```mermaid
flowchart LR
    A["delete temp directory<br/><i>type: action<br/>freq: 1</i>"] -->|freq 1| F["build cache wipe<br/><i>type: feedback<br/>freq: 1</i>"]
    F -->|freq 1| O["build take 40 minute<br/><i>type: outcome<br/>freq: 1</i>"]

    style A fill:#ffe0e0,stroke:#c00
    style F fill:#fff4e0,stroke:#c80
    style O fill:#e0e0ff,stroke:#00c
```

### Step 2 — A *differently worded* version of the same mistake

```python
guard.record(
    action="clear out /tmp before the run",
    feedback="cache directory got removed",
    outcome="next build took 40 minutes",
)
```

The merge engine embeds each piece and compares it to existing nodes of the same type. Similarity beats the threshold, so **no new nodes are created** — the existing ones increment:

```mermaid
flowchart LR
    A["delete temp directory<br/><i>freq: 2 ⬆</i>"] -->|freq 2 ⬆| F["build cache wipe<br/><i>freq: 2 ⬆</i>"]
    F -->|freq 2 ⬆| O["build take 40 minute<br/><i>freq: 2 ⬆</i>"]

    style A fill:#ffcccc,stroke:#c00,stroke-width:3px
    style F fill:#ffe8cc,stroke:#c80,stroke-width:3px
    style O fill:#ccccff,stroke:#00c,stroke-width:3px
```

This is the core idea: **paraphrases collapse into one memory and make it stronger**, instead of scattering into near-duplicate entries.

### Step 3 — The agent proposes it a third time

```python
review = guard.before_step("remove everything under the tmp folder")
```

Retrieval embeds the query, finds the nearest `action` nodes, walks each causal chain forward, and returns:

```python
review.warnings[0].trigger      # "delete temp directory"
review.warnings[0].risk         # "build cache wipe"
review.warnings[0].frequency    # 2
review.warnings[0].confidence   # 0.79
review.chains[0]                # ["delete temp directory", "build cache wipe", "build take 40 minute"]
```

The agent now has a concrete, cited reason to revise the step — before anything is executed.

> **Note on what you see:** `trigger` and `risk` are the *normalized* node texts, not your original sentences. `risk` is the **feedback** node (the immediate signal), while the full `action → feedback → outcome` path is available in `review.chains`.

---

## High-Level Design (HLD)

### System context

```mermaid
flowchart TB
    subgraph clients["Client Layer — pick any entrypoint"]
        direction LR
        C1["Python agent<br/>DriftGuard()"]
        C2["MCP client<br/>Claude Desktop, etc."]
        C3["LangGraph<br/>review node"]
        C4["Any planner<br/>review_payload()"]
    end

    subgraph facade["Facade Layer"]
        GUARD["DriftGuard<br/>policy enforcement"]
    end

    subgraph core["Core Layer"]
        RT["DriftGuardRuntime<br/>orchestration"]
    end

    subgraph engines["Engine Layer"]
        direction LR
        ME["MergeEngine<br/>dedupe"]
        RE["RetrievalEngine<br/>rank"]
        PE["PruneEngine<br/>hygiene"]
        EE["EmbeddingEngine<br/>vectors"]
    end

    subgraph memory["Memory Layer"]
        direction LR
        GM["GraphStore<br/>mistakes"]
        GS["GraphStore<br/>successes"]
    end

    subgraph storage["Persistence Layer"]
        direction LR
        J[(JSON)]
        S[(SQLite)]
        P[(Postgres)]
    end

    C1 --> GUARD
    C3 --> GUARD
    C4 --> GUARD
    C2 --> RT
    GUARD --> RT
    RT --> ME & RE & PE
    ME --> EE
    RT --> GM & GS
    GM --> J & S & P
    GS --> J & S & P

    style GUARD fill:#fff3cd,stroke:#856404,stroke-width:2px
    style RT fill:#d1ecf1,stroke:#0c5460,stroke-width:2px
```

### Layer responsibilities

| Layer | Component | Owns |
| --- | --- | --- |
| Client | your code | Deciding what to do with warnings |
| Facade | `DriftGuard` | Policy (warn / block / acknowledge / record_only), thresholds |
| Core | `DriftGuardRuntime` | Wiring, dual-graph coordination, metrics, save/load |
| Engine | `MergeEngine` | Normalize, embed, decide "is this the same node?" |
| Engine | `RetrievalEngine` | Similarity search, chain walk, confidence, dedupe, sort |
| Engine | `PruneEngine` | Removing weak edges, stale nodes, isolates |
| Engine | `EmbeddingEngine` | sentence-transformers wrapper, swappable |
| Memory | `GraphStore` | The `networkx.DiGraph` itself, node/edge lifecycle |
| Persistence | `GraphPersistence` | Bytes on disk / rows in a database |

### The dual-graph design

DriftGuard keeps **two completely independent graphs**:

```mermaid
flowchart LR
    Q(["query:<br/>'add more salt'"]) --> M["Mistake graph"]
    Q --> S["Success graph"]
    M --> W["warnings[]"]
    S --> R["reinforcements[]"]
    W --> RESP["RetrievalResponse"]
    R --> RESP

    style M fill:#ffe0e0,stroke:#c00
    style S fill:#e0ffe0,stroke:#0a0
```

They share engine *instances* but never share nodes, edges, or storage. Recording a mistake can never weaken a success memory. Both are queried on every `before_step`, so the agent hears "this failed before" and "this worked before" in the same breath.

### Request flow — the two operations

```mermaid
sequenceDiagram
    autonumber
    participant A as Agent
    participant G as DriftGuard
    participant RT as Runtime
    participant RE as RetrievalEngine
    participant GS as GraphStore
    participant M as Metrics

    Note over A,M: READ PATH — before_step()
    A->>G: before_step("remove tmp folder")
    G->>G: resolve policy + min_confidence
    alt policy == record_only
        G-->>A: empty RetrievalResponse (review skipped)
    else
        G->>RT: query_memory(context)
        RT->>RE: query(context)
        RE->>GS: find_similar_nodes(type="action", top_k)
        GS-->>RE: [(node, similarity), ...]
        loop each matched node
            RE->>GS: get_related_chains(node)
            GS-->>RE: [[action, feedback, outcome], ...]
        end
        RE->>RE: score → dedupe → sort
        RE->>M: record_review(...)
        RE-->>RT: RetrievalResponse
        RT-->>G: RetrievalResponse
        alt policy == block AND confidence ≥ min
            G--xA: raise GuardrailTriggered
        else policy == acknowledge AND not acknowledged
            G--xA: raise GuardrailAcknowledgementRequired
        else
            G-->>A: RetrievalResponse
        end
    end

    Note over A,M: WRITE PATH — record()
    A->>G: record(action, feedback, outcome)
    G->>RT: register_mistake(...)
    RT->>GS: add_event(Event)
    GS->>GS: get_or_create × 3, add_edge × 2
    GS->>GS: light_prune (no-op by design)
    RT->>GS: save()
    RT->>M: record_storage()
    RT-->>A: {"status": "stored", ...}
```

---

## Low-Level Design (LLD)

### Class structure

```mermaid
classDiagram
    class DriftGuard {
        +runtime: DriftGuardRuntime
        +settings: DriftGuardSettings
        +before_step(context, policy, min_confidence, acknowledged) RetrievalResponse
        +review(context) RetrievalResponse
        +record(action, feedback, outcome) dict
        +record_success(action, feedback, outcome) dict
        +prune() dict
        +stats() dict
        -_should_block()
        -_should_acknowledge()
        -_resolve_policy()
    }

    class DriftGuardRuntime {
        +settings
        +graph_store: GraphStore
        +success_graph_store: GraphStore
        +retrieval_engine: RetrievalEngine
        +metrics: DriftGuardMetrics
        +register_mistake() dict
        +register_success() dict
        +query_memory(context) RetrievalResponse
        +deep_prune() dict
        +graph_stats() dict
        +metrics_snapshot() dict
    }

    class GraphStore {
        +graph: nx.DiGraph
        +add_event(event)
        +find_similar_nodes(text, node_type, top_k, min_similarity)
        +get_related_chains(node_text, depth, max_branching, max_paths)
        +get_node(node) dict
        +stats() dict
        +save()
        +load()
        -_get_or_create_node(text, node_type)
        -_create_node(text, node_type)
        -_add_edge(src, dst)
    }

    class MergeEngine {
        +normalize(text) str
        +embed(text) ndarray
        +find_similar_node(text, node_type, graph) str
        +find_top_k_similar(text, graph, node_type, top_k) list
        -_get_threshold(node_type) float
    }

    class RetrievalEngine {
        +top_k: int
        +min_similarity: float
        +recency_weight: float
        +query(context) RetrievalResponse
        -_collect_chain_matches()
        -_build_signals()
        -_confidence()
        -_recency_score()
    }

    class PruneEngine {
        +node_stale_days: int
        +edge_min_frequency: int
        +light_prune(graph)
        +deep_prune(graph) dict
        -_remove_weak_edges()
        -_remove_stale_nodes()
        -_remove_isolated_nodes()
    }

    class EmbeddingEngine {
        +embed(text) ndarray
        +embed_batch(texts) ndarray
        +model_name() str
    }

    class GraphPersistence {
        <<Protocol>>
        +save_graph(graph)
        +load_graph() DiGraph
    }

    DriftGuard --> DriftGuardRuntime
    DriftGuardRuntime --> GraphStore : mistakes + successes
    DriftGuardRuntime --> RetrievalEngine
    DriftGuardRuntime --> PruneEngine
    GraphStore --> MergeEngine
    GraphStore --> PruneEngine
    GraphStore --> GraphPersistence
    MergeEngine --> EmbeddingEngine
    RetrievalEngine --> GraphStore

    GraphPersistence <|.. Persistence
    GraphPersistence <|.. SQLitePersistence
    GraphPersistence <|.. PostgresPersistence
```

### LLD 1 — Node merge decision

Every piece of text entering the graph goes through this. It is what makes "delete the temp directory" and "clear out /tmp" the same memory.

```mermaid
flowchart TD
    START([text + node_type]) --> N["normalize_text via spaCy<br/>lowercase, lemmatize, drop stopwords"]
    N --> C{"any existing nodes<br/>of this type?"}
    C -->|no| CREATE
    C -->|yes| EMB["embed normalized text<br/>MiniLM, L2-normalized"]
    EMB --> LOOP["cosine similarity vs<br/>every candidate node"]
    LOOP --> BEST{"best score ><br/>type threshold?"}
    BEST -->|yes| MERGE["reuse node<br/>frequency += 1<br/>last_seen = now<br/>metrics.record_node_merged"]
    BEST -->|no| CREATE["create node<br/>frequency = 1<br/>first_seen = last_seen = now<br/>metrics.record_node_created"]
    MERGE --> END([canonical node key])
    CREATE --> END

    style MERGE fill:#d4edda,stroke:#155724
    style CREATE fill:#f8d7da,stroke:#721c24
```

**Thresholds are per node type, and deliberately asymmetric:**

| Node type | Default threshold | Why |
| --- | --- | --- |
| `action` | `0.72` | Actions are phrased many ways; merge generously |
| `feedback` | `0.70` | Error messages vary in wording; merge generously |
| `outcome` | `0.88` | Strict — conflating two different consequences is the costly error |

A node's **key is its normalized text**, so the graph is naturally deduplicated at the string level before embeddings are ever consulted.

### LLD 2 — Retrieval pipeline

```mermaid
flowchart TD
    Q([context string]) --> TK["find_top_k_similar<br/>node_type = action"]
    TK --> VEC["stack node embeddings into matrix<br/>scores = E · q  (vectorized)"]
    VEC --> FILT{"score ≥<br/>min_similarity?"}
    FILT -->|no| DROP[discard]
    FILT -->|yes| WALK["get_related_chains<br/>DFS forward"]
    WALK --> LIM["bounded by:<br/>max_depth = 3<br/>max_branching = 10<br/>max_paths = 100"]
    LIM --> SIG["for each chain:<br/>trigger = chain[0]<br/>risk = chain[1]"]
    SIG --> CONF["confidence =<br/>f(node_freq, edge_freq, similarity, recency)"]
    CONF --> DEDUP["dedupe on (trigger, second)<br/>keep highest confidence"]
    DEDUP --> SORT["sort by confidence desc"]
    SORT --> OUT([RetrievalResponse])

    style CONF fill:#fff3cd,stroke:#856404
```

The same pipeline runs against the success graph to produce `reinforcements`. Response-level `confidence` is the **maximum warning confidence**, which is what the block/acknowledge policies compare against.

Chain traversal is bounded on three axes so a densely connected graph can't cause a traversal blowup: depth, branching per node, and total paths. Neighbors are visited **highest edge-frequency first**, so if a cap truncates the search, you keep the strongest causal links.

### LLD 3 — Prune ordering

Order matters here, and the code is explicit about it:

```mermaid
flowchart LR
    S([deep_prune]) --> E["1. remove weak edges<br/>frequency < edge_min_frequency"]
    E --> N["2. remove stale nodes<br/>last_seen older than node_stale_days"]
    N --> I["3. remove isolated nodes<br/>in_degree = 0 and out_degree = 0"]
    I --> SAVE["4. save both graphs"]
    SAVE --> R([summary dict])

    style E fill:#ffe0e0
    style N fill:#fff0e0
    style I fill:#e0e0ff
```

Weak edges are removed first *because* doing so exposes newly isolated nodes for step 3. Running the steps in any other order leaves orphans behind.

`light_prune()` is called after every insert and is an intentional no-op — a hook reserved for cheap guards like a hard node cap. Keep expensive work in `deep_prune()`, which you call on a schedule or via the `deep_prune` MCP tool.

### LLD 4 — Guard policy state machine

```mermaid
stateDiagram-v2
    [*] --> Resolve: before_step(context)
    Resolve --> Skipped: policy == record_only
    Resolve --> Query: otherwise
    Skipped --> [*]: empty response

    Query --> NoMatch: no warnings
    Query --> Matched: warnings found

    NoMatch --> [*]: response returned

    Matched --> Warn: policy == warn
    Matched --> Block: policy == block AND confidence ≥ min
    Matched --> Ack: policy == acknowledge AND NOT acknowledged

    Warn --> [*]: response returned
    Block --> [*]: raise GuardrailTriggered
    Ack --> [*]: raise GuardrailAcknowledgementRequired
```

Note that `raise_on_match=True` promotes `warn` (or an unset policy) to `block` for that single call — useful for guarding one critical step without changing global settings.

---

## Data Model

### Node

| Attribute | Type | Notes |
| --- | --- | --- |
| *key* | `str` | The normalized text — this **is** the node identity |
| `type` | `"action" \| "feedback" \| "outcome"` | Constrains merge and retrieval |
| `embedding` | `np.ndarray` | L2-normalized, so dot product = cosine similarity |
| `frequency` | `int` | Times this node was seen or merged into |
| `first_seen` | `datetime` (UTC) | Set at creation |
| `last_seen` | `datetime` (UTC) | Bumped on every merge; drives recency and staleness |

### Edge

| Attribute | Type | Notes |
| --- | --- | --- |
| `frequency` | `int` | Times this exact causal link was observed |
| `weight` | `float` | Reserved for future weighting |
| `created_at` | `datetime` (UTC) | Set at creation |

### Response objects

```python
@dataclass
class Warning:
    trigger: str       # normalized action node
    risk: str          # normalized feedback node
    frequency: int     # trigger node frequency
    confidence: float

@dataclass
class Reinforcement:
    trigger: str
    recommendation: str
    frequency: int
    confidence: float

@dataclass
class RetrievalResponse:
    query: str
    warnings: list[Warning]
    chains: list[list[str]]          # full action → feedback → outcome paths
    confidence: float                # max warning confidence
    reinforcements: list[Reinforcement]
    timestamp: datetime
```

---

## Confidence Scoring, Explained

Confidence answers: *how much should the agent trust this warning?* It blends three signals.

```
combined = (node_frequency + edge_frequency) / 2

           ┌ combined ≥ 5  →  0.95
reinforce ─┤ combined ≥ 3  →  0.85
           │ combined ≥ 2  →  0.75
           └ else          →  0.60

           ┌ ≤ 1 day       →  1.00
recency   ─┤ ≤ 7 days      →  0.85
           │ ≤ 30 days     →  0.70
           └ else          →  0.50

remaining = 1 − recency_weight

confidence = 0.65·remaining·reinforce
           + 0.35·remaining·similarity
           + recency_weight·recency
```

capped at `1.0`. With no usable timestamp, it falls back to `0.65·reinforce + 0.35·similarity`.

**Why edge frequency and not just node frequency:** a node seen 10 times tells you the *action* is common. An edge seen 10 times tells you that action *reliably causes that feedback*. The edge is the stronger causal claim, so both are averaged rather than using node count alone.

**Worked number** — a mistake seen twice, matched at similarity `0.81`, last seen yesterday, default `recency_weight=0.15`:

```
combined  = (2 + 2) / 2 = 2      → reinforce = 0.75
recency   = 1.0
remaining = 0.85

confidence = 0.65·0.85·0.75 + 0.35·0.85·0.81 + 0.15·1.0
           = 0.414 + 0.241 + 0.150
           = 0.81
```

---

## Guard Policies

| Policy | Behavior | Use when |
| --- | --- | --- |
| `warn` | Return warnings, let the agent decide | Default; planner is capable of revising |
| `block` | Raise `GuardrailTriggered` | Irreversible steps — deletes, payments, deploys |
| `acknowledge` | Raise `GuardrailAcknowledgementRequired` until confirmed | Human-in-the-loop gates |
| `record_only` | Skip review entirely, still record | Warm-up runs, memory-building passes |

```python
from driftguard import DriftGuard, DriftGuardSettings

guard = DriftGuard(
    settings=DriftGuardSettings(
        guard_policy="acknowledge",
        guard_min_confidence=0.8,   # only gate on strong matches
    )
)
```

`guard_min_confidence` is the escape hatch that keeps `block` usable: low-confidence recollections pass through as ordinary warnings, and only well-reinforced memories stop the run.

---

## Integrations

### MCP server

```bash
driftguard-mcp
```

```json
{
  "mcpServers": {
    "driftguard": {
      "command": "driftguard-mcp"
    }
  }
}
```

| Tool | Purpose |
| --- | --- |
| `register_mistake` | Store an `action → feedback → outcome` failure |
| `register_success` | Store the same shape into the success graph |
| `query_memory` | Return `warnings` + `reinforcements` for a context |
| `deep_prune` | Run full cleanup on both graphs |
| `graph_stats` | Node/edge counts for both graphs |
| `guard_metrics` | Runtime counters and gauges |

### LangGraph

```python
from driftguard import DriftGuard, make_langgraph_review_node

review_node = make_langgraph_review_node(
    guard,
    action_key="candidate_action",     # which state key holds the action
    policy="warn",
)

workflow.add_node("guard_review", review_node)
```

The node reads `state[action_key]` and writes back `guard_review`, `guard_warnings_count`, `guard_confidence`, `guard_top_warning`, `guard_reinforcements_count`, and `guard_top_reinforcement` — all key names configurable.

### Any planner

```python
from driftguard import DriftGuard, review_payload

result = review_payload(guard, {"action": "drop the staging table", "attempt": 2})
result["warnings_count"], result["confidence"]
```

### Decorator

```python
from driftguard import guard_step

@guard_step(guard, raise_on_match=True)
def execute(action: str):
    ...
```

The context is taken from the first string argument, or from an `action` / `context` / `task` / `prompt` keyword — or supply your own `input_getter`.

---

## Storage Backends

```mermaid
flowchart LR
    GS["GraphStore"] --> PROTO["GraphPersistence<br/><i>Protocol</i>"]
    PROTO --> J["Persistence<br/>JSON + node_link"]
    PROTO --> S["SQLitePersistence<br/>nodes / edges tables"]
    PROTO --> P["PostgresPersistence<br/>SQLAlchemy + psycopg"]

    style PROTO fill:#d1ecf1,stroke:#0c5460
```

| Backend | Best for | Format |
| --- | --- | --- |
| JSON *(default)* | Local experiments, inspectable state | `networkx` node-link, atomic write via temp + `os.replace` |
| SQLite | Single-host production | `nodes` / `edges` tables, full rewrite in one transaction |
| Postgres | Multi-process / shared | `driftguard_meta`, `_nodes`, `_edges`; embeddings as JSONB |

JSON was chosen over pickle deliberately: human-readable, safe across Python versions, and no arbitrary-code-execution risk on load.

The two graphs persist independently:

| Backend | Mistake graph | Success graph |
| --- | --- | --- |
| JSON | `driftguard_graph.json` | `driftguard_success_graph.json` |
| SQLite | `driftguard_graph.sqlite3` | `driftguard_success_graph.sqlite3` |
| Postgres | `driftguard_*` tables | `success_driftguard_*` tables |

```python
from driftguard import DriftGuardSettings

settings = DriftGuardSettings(
    storage_backend="postgres",
    postgres_dsn="postgresql+psycopg://user:pass@host:5432/driftguard",
)
```

> **Scaling note:** saves are full-graph rewrites and similarity search is a linear scan over in-memory embeddings. This is fine for thousands of nodes. If you're heading for hundreds of thousands, you want an ANN index and incremental writes — not yet implemented.

---

## Configuration Reference

Every field on `DriftGuardSettings` (frozen dataclass):

| Setting | Default | Controls |
| --- | --- | --- |
| `storage_backend` | `"json"` | `json` / `sqlite` / `postgres` |
| `graph_filepath` | `"driftguard_graph.json"` | JSON mistake graph path |
| `success_graph_filepath` | `"driftguard_success_graph.json"` | JSON success graph path |
| `sqlite_filepath` | `"driftguard_graph.sqlite3"` | SQLite mistake graph path |
| `success_sqlite_filepath` | `"driftguard_success_graph.sqlite3"` | SQLite success graph path |
| `postgres_dsn` | `None` | Postgres connection string |
| `embedding_model_name` | `"sentence-transformers/all-MiniLM-L6-v2"` | Embedding model |
| `embedding_device` | `None` | `cpu`, `cuda`, etc. |
| `retrieval_top_k` | `5` | Action nodes considered per query |
| `retrieval_min_similarity` | `0.60` | Floor for a match to count |
| `retrieval_recency_weight` | `0.15` | Recency share of confidence, clamped to `0.5` |
| `traversal_max_depth` | `3` | Chain walk depth |
| `traversal_max_branching` | `10` | Neighbors expanded per node |
| `traversal_max_paths` | `100` | Total paths per query |
| `similarity_threshold_action` | `0.72` | Action merge threshold |
| `similarity_threshold_feedback` | `0.70` | Feedback merge threshold |
| `similarity_threshold_outcome` | `0.88` | Outcome merge threshold |
| `guard_policy` | `"warn"` | Default policy |
| `guard_min_confidence` | `0.0` | Floor for block / acknowledge |
| `prune_node_stale_days` | `60` | Staleness window |
| `prune_edge_min_frequency` | `2` | Edges below this are pruned |
| `log_level` | `"INFO"` | Logging verbosity |

### Tuning guide

| Symptom | Try |
| --- | --- |
| Too many false warnings | Raise `retrieval_min_similarity` toward `0.70` |
| Real repeats not caught | Lower `similarity_threshold_action` toward `0.65` |
| Near-duplicate nodes piling up | Lower the merge thresholds |
| Distinct failures collapsing into one | Raise `similarity_threshold_outcome` |
| Old memories dominating | Raise `retrieval_recency_weight` toward `0.3` |
| Graph growing unbounded | Lower `prune_node_stale_days`, schedule `deep_prune()` |

---

## Metrics Reference

```python
from driftguard import build_runtime

runtime = build_runtime()
snapshot = runtime.metrics_snapshot()
print(snapshot["counters"], snapshot["gauges"])
```

**Counters:** `reviews_total`, `reviews_skipped_total`, `reviews_blocked_total`, `reviews_ack_required_total`, `review_warnings_total`, `records_total`, `nodes_created_total`, `nodes_merged_total`, `edges_created_total`, `edges_reused_total`, `prune_runs_total`, `prune_nodes_removed_total`, `prune_edges_removed_total`

**Gauges:** `last_review_confidence`, `review_confidence_average`, `review_confidence_total`

The ratio `nodes_merged_total / nodes_created_total` is the single most useful number here: it tells you whether your merge thresholds are actually collapsing paraphrases. Near zero means every phrasing is creating a new node and the memory isn't generalizing.

---

## Benchmarks and Honest Limitations

### Running the benchmark

```bash
driftguard-benchmark
driftguard-benchmark --format json
```

It reports merge precision/recall, retrieval precision/recall, and F1.

### What the benchmark does and does not prove

**Read this before quoting any number from it.** The built-in suite is a *regression test for the graph logic*, not a measure of real-world quality:

- It runs on **4 seed events and 7 cases**, all from one cooking scenario.
- It substitutes a hand-built `BenchmarkEmbeddingEngine` — a bag-of-features vector with a hardcoded alias table (`salty → salt`, `charred → burn`) — in place of the real sentence-transformers model.
- Thresholds are lowered specifically for that stub embedder.

So a strong score confirms that merge and retrieval behave correctly *given a well-behaved embedding space*. It says nothing about whether MiniLM at `0.72` separates your agent's actions correctly. The unit tests likewise use `StubEmbeddingEngine` with hand-written vectors.

**There is currently no evaluation of the default configuration on real agent traces.** If you adopt DriftGuard, plan to tune the thresholds against your own data, and treat the defaults as a starting point rather than a validated setting. Contributions of a real trace-based eval are the single most valuable thing anyone could add to this project.

### Known limitations

| Limitation | Impact |
| --- | --- |
| Query text is embedded **raw**, node text is embedded **normalized** | Slight asymmetry between the query vector and stored vectors; may cost recall |
| Linear scan over all nodes of a type | Fine to ~thousands of nodes, not to hundreds of thousands |
| Full-graph rewrite on every save | Write cost grows with graph size |
| `sentence-transformers` and `spacy` are hard dependencies | Multi-GB install for a guardrail layer |
| English-only normalization | `en_core_web_sm` is the only supported spaCy model |
| Single-process assumption | Concurrent writers to the same JSON/SQLite file will clobber each other; use Postgres for shared deployments |
| Warnings surface **normalized** text | Displayed triggers won't match your original phrasing verbatim |

---

## Demos

**Deterministic, offline, no API key:**

```bash
python demo/rule_based/demo_agent.py
```

Walks through merge behavior, warning retrieval, pruning, and graph evolution with printed state at each step. This is the fastest way to see the system work.

**Live LLM agent:**

```bash
pip install "driftguard-ai[demo]"
python demo/langgraph/demo_agent.py
```

A full planner → guard → revise → execute loop in LangGraph.

---

## Project Status

**Beta (`0.2.0`).** Implemented and tested:

- Semantic merge engine with per-type thresholds
- Similarity retrieval with chain traversal and confidence scoring
- Dual mistake/success graphs with positive reinforcement
- JSON, SQLite, and Postgres persistence
- MCP server (6 tools) and LangGraph adapter
- Prune engine, runtime metrics, benchmark harness
- ~2,500 lines of source against a comparable volume of tests, CI on every push

**Honest positioning:** suitable for experimentation and agent-infrastructure research. The defaults have not been validated against real agent traces — see [Benchmarks and Honest Limitations](#benchmarks-and-honest-limitations) before relying on it in production.

**Roadmap:**

- Evaluation on real agent failure traces
- Pluggable embedding backends (OpenAI, local GGUF) so PyTorch becomes optional
- ANN index for large graphs
- Incremental persistence
- Normalize query text on the retrieval path to match node embeddings

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

```bash
git clone https://github.com/sujal-maheshwari2004/DriftGuard
cd DriftGuard
pip install -e ".[dev]"
python -m spacy download en_core_web_sm
pytest
```

Most wanted, in order:

1. A real-trace evaluation harness
2. Embedding backends that don't require PyTorch
3. Adapters for other agent frameworks
4. Non-English normalization

---

## License

MIT — see [LICENSE](LICENSE).
