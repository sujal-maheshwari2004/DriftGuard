from DriftGuard.models.event import Event

from DriftGuard.graph.graph_store import GraphStore
from DriftGuard.graph.merge_engine import MergeEngine

from DriftGuard.graph.prune_engine import PruneEngine

from DriftGuard.retrieval.retrieval_engine import RetrievalEngine


# =====================================================
# Dummy persistence layer for test-only environment
# =====================================================

class DummyPersistenceEngine:

    def save_graph(self, graph):

        pass

    def load_graph(self):

        return None


# =====================================================
# INITIALIZE CORE COMPONENTS
# =====================================================

merge_engine = MergeEngine()

prune_engine = PruneEngine()

persistence_engine = DummyPersistenceEngine()


graph_store = GraphStore(

    merge_engine,

    prune_engine,

    persistence_engine,
)


retriever = RetrievalEngine(graph_store)


# =====================================================
# INSERT TEST EVENTS
# =====================================================

events = [

    Event(

        action="increase salt",

        feedback="too salty",

        outcome="dish ruined",
    ),

    Event(

        action="add more salt",

        feedback="over-seasoned",

        outcome="dish ruined",
    ),
]


for event in events:

    graph_store.add_event(event)


# =====================================================
# RUN RETRIEVAL TEST
# =====================================================

result = retriever.query("increase salt")


print("\n========== DRIFTGUARD PIPELINE TEST ==========\n")

print("Query:\n")

print(result.query)


print("\nWarnings:\n")

for warning in result.warnings:

    print(warning)


print("\nChains:\n")

for chain in result.chains:

    print(chain)


print("\nConfidence:\n")

print(result.confidence)


print("\nGraph Stats:\n")

print(graph_store.stats())