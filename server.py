from pathlib import Path
import sys


SRC_DIR = Path(__file__).resolve().parent / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def main():
    from driftguard.logging_config import configure_logging
    from driftguard.server import main as run_server

    configure_logging()
    run_server()


if __name__ == "__main__":
    main()
