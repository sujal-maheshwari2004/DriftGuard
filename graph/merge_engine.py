import pickle
from pathlib import Path
from datetime import datetime


class PersistenceEngine:
    """
    Handles DriftGuard graph persistence lifecycle.

    Responsibilities:
    - save graph to disk
    - load graph from disk
    - checkpoint memory state
    - enable restart recovery

    Current backend:
        pickle (fast + NetworkX compatible)

    Future upgrade targets:
        SQLite
        Neo4j
        DuckDB
        Parquet snapshots
    """

    def __init__(
        self,
        filepath="driftguard_graph.pkl",
        autosave=True,
    ):
        """
        filepath:
            location of graph snapshot file

        autosave:
            automatically persist after save()
        """

        self.filepath = Path(filepath)

        self.autosave = autosave

    # =====================================================
    # SAVE GRAPH
    # =====================================================

    def save_graph(self, graph):
        """
        Persist graph to disk.
        """

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
        """
        Restore graph from disk.
        """

        if not self.filepath.exists():

            return None

        with open(self.filepath, "rb") as f:

            return pickle.load(f)

    # =====================================================
    # CHECKPOINT SNAPSHOT
    # =====================================================

    def checkpoint(self, graph):
        """
        Save timestamped backup snapshot.
        """

        timestamp = datetime.now().strftime(
            "%Y%m%d_%H%M%S"
        )

        backup_path = self.filepath.with_name(
            f"{self.filepath.stem}_{timestamp}.pkl"
        )

        with open(backup_path, "wb") as f:

            pickle.dump(graph, f)

    # =====================================================
    # EXISTS CHECK
    # =====================================================

    def exists(self):
        """
        Check whether graph snapshot exists.
        """

        return self.filepath.exists()