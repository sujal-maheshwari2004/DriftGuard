import pickle
from pathlib import Path


class Persistence:
    """
    Handles saving and loading DriftGuard graph memory.
    """

    def __init__(self, filepath="driftguard_graph.pkl"):

        self.filepath = Path(filepath)

    # =====================================================
    # SAVE GRAPH
    # =====================================================

    def save_graph(self, graph):

        self.filepath.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(self.filepath, "wb") as f:

            pickle.dump(graph, f)

    # =====================================================
    # LOAD GRAPH
    # =====================================================

    def load_graph(self):

        if not self.filepath.exists():

            return None

        with open(self.filepath, "rb") as f:

            return pickle.load(f)