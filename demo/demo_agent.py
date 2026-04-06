"""Compatibility wrapper for the canonical rule-based demo."""

from pathlib import Path
import sys


ROOT_DIR = Path(__file__).resolve().parents[1]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


from demo.rule_based.demo_agent import *  # noqa: F401,F403
from demo.rule_based.demo_agent import main


if __name__ == "__main__":
    main()
