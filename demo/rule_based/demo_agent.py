from __future__ import annotations

from argparse import ArgumentParser, Namespace
from dataclasses import asdict, dataclass
from pathlib import Path
import json
import sys
import time

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from driftguard import DriftGuard, DriftGuardSettings, configure_logging
from driftguard.graph.merge_engine import MergeEngine
from driftguard.runtime import build_runtime


DEFAULT_GRAPH_PATH = (
    ROOT_DIR / "demo" / "output" / "rule_based" / "kitchen_line_graph.json"
)
DEFAULT_TRACE_PATH = (
    ROOT_DIR / "demo" / "output" / "rule_based" / "kitchen_line_trace.jsonl"
)


@dataclass(frozen=True)
class RiskEvent:
    action: str
    feedback: str
    outcome: str


@dataclass(frozen=True)
class ScenarioFamily:
    name: str
    risky_actions: tuple[str, ...]
    feedbacks: tuple[str, ...]
    outcomes: tuple[str, ...]
    safe_action: str
    safe_reason: str


@dataclass(frozen=True)
class StepPlan:
    phase: str
    family_name: str
    intent: str
    risky_event: RiskEvent
    safe_action: str
    safe_reason: str


SCENARIO_FAMILIES: tuple[ScenarioFamily, ...] = (
    ScenarioFamily(
        name="seasoning",
        risky_actions=(
            "increase salt",
            "add more salt",
            "season more aggressively",
            "finish with extra salt",
        ),
        feedbacks=(
            "too salty",
            "over-seasoned",
            "salt overwhelms the dish",
            "salinity is too high",
        ),
        outcomes=(
            "dish ruined",
            "plate sent back",
            "order had to be remade",
            "guest complained about seasoning",
        ),
        safe_action="taste before adding more salt",
        safe_reason="verify the baseline flavor before changing seasoning",
    ),
    ScenarioFamily(
        name="heat-control",
        risky_actions=(
            "raise pan heat",
            "cook on higher heat",
            "blast the burner hotter",
            "finish on maximum heat",
        ),
        feedbacks=(
            "outside burned before center cooked",
            "surface charred too fast",
            "pan scorched the protein",
            "heat was too aggressive",
        ),
        outcomes=(
            "protein was unusable",
            "ticket was delayed for a recook",
            "dish came out burned",
            "guest received a charred entree",
        ),
        safe_action="lower heat and extend cook time",
        safe_reason="gentler heat protects the center while the outside browns",
    ),
    ScenarioFamily(
        name="oil-management",
        risky_actions=(
            "add more oil",
            "grease the pan heavily",
            "pour extra oil into the skillet",
            "coat the pan with more fat",
        ),
        feedbacks=(
            "dish turned greasy",
            "sauce separated from excess oil",
            "texture felt oily",
            "plate looked slick with fat",
        ),
        outcomes=(
            "customer disliked the texture",
            "dish needed to be restarted",
            "finished plate looked unappetizing",
            "guest sent the dish back",
        ),
        safe_action="measure a small amount of oil first",
        safe_reason="measured fat gives control without making the dish greasy",
    ),
    ScenarioFamily(
        name="resting",
        risky_actions=(
            "plate immediately after cooking",
            "skip the resting time",
            "cut the protein right away",
            "serve without letting it rest",
        ),
        feedbacks=(
            "juices ran onto the board",
            "meat dried out quickly",
            "texture tightened after slicing",
            "protein lost moisture during plating",
        ),
        outcomes=(
            "dish tasted dry",
            "guest noticed the meat was dry",
            "protein quality dropped on the pass",
            "plate quality fell below standard",
        ),
        safe_action="rest the protein before plating",
        safe_reason="resting keeps moisture in the final dish",
    ),
)


NOISE_EVENTS: tuple[RiskEvent, ...] = (
    RiskEvent(
        action="skip labeling the sauce bottle",
        feedback="another cook could not identify the sauce",
        outcome="station setup slowed down",
    ),
    RiskEvent(
        action="stack hot pans on the prep board",
        feedback="prep board heated up unexpectedly",
        outcome="mise en place had to be moved",
    ),
    RiskEvent(
        action="leave the garnish tray uncovered",
        feedback="garnish wilted faster than expected",
        outcome="fresh garnish had to be replaced",
    ),
)

DEMO_FEATURES: tuple[str, ...] = (
    "salt",
    "taste",
    "heat",
    "burn",
    "oil",
    "grease",
    "rest",
    "plate",
    "protein",
    "dry",
    "complaint",
    "delay",
    "sauce",
    "label",
    "board",
    "garnish",
    "replace",
)

DEMO_TOKEN_ALIASES: dict[str, str] = {
    "adding": "add",
    "aggressively": "aggressive",
    "board": "board",
    "burned": "burn",
    "burner": "heat",
    "burning": "burn",
    "charred": "burn",
    "coat": "oil",
    "cooking": "cook",
    "complained": "complaint",
    "complaints": "complaint",
    "cook": "heat",
    "cooked": "heat",
    "cutting": "plate",
    "drying": "dry",
    "fat": "oil",
    "fatty": "grease",
    "finished": "finish",
    "garnishes": "garnish",
    "greasy": "grease",
    "grease": "oil",
    "greased": "oil",
    "guest": "complaint",
    "guests": "complaint",
    "heavily": "heavy",
    "hot": "heat",
    "hotter": "heat",
    "juices": "dry",
    "labeling": "label",
    "maximum": "heat",
    "moisture": "dry",
    "oil": "oil",
    "oily": "grease",
    "order": "delay",
    "orders": "delay",
    "pan": "heat",
    "plated": "plate",
    "plating": "plate",
    "protein": "protein",
    "resting": "rest",
    "salinity": "salt",
    "salted": "salt",
    "salting": "salt",
    "salty": "salt",
    "sauce": "sauce",
    "season": "salt",
    "seasoned": "salt",
    "seasoning": "salt",
    "serve": "plate",
    "serving": "plate",
    "slicing": "plate",
    "supervisor": "complaint",
    "taste": "taste",
    "tasting": "taste",
    "ticket": "delay",
    "uncovered": "garnish",
    "wilted": "garnish",
}

DEMO_STOPWORDS = {
    "a",
    "after",
    "and",
    "before",
    "for",
    "immediately",
    "into",
    "it",
    "let",
    "letting",
    "more",
    "of",
    "on",
    "the",
    "time",
    "to",
    "too",
    "up",
    "with",
    "without",
}


def simple_normalize_text(text: str) -> str:
    cleaned = "".join(
        character.lower() if character.isalnum() else " "
        for character in text
    )
    tokens = []

    for token in cleaned.split():
        if token in DEMO_STOPWORDS:
            continue

        tokens.append(DEMO_TOKEN_ALIASES.get(token, token))

    return " ".join(tokens)


class DemoEmbeddingEngine:
    def __init__(self):
        self._feature_index = {
            feature: index for index, feature in enumerate(DEMO_FEATURES)
        }
        self._hash_buckets = 4

    def embed(self, text: str) -> np.ndarray:
        normalized = simple_normalize_text(text)
        vector = np.zeros(
            len(self._feature_index) + self._hash_buckets,
            dtype=np.float32,
        )

        for token in normalized.split():
            index = self._feature_index.get(token)

            if index is not None:
                vector[index] += 1.0
            else:
                bucket = sum(ord(character) for character in token) % self._hash_buckets
                vector[len(self._feature_index) + bucket] += 0.25

        if not vector.any():
            vector[-1] = 1.0

        norm = np.linalg.norm(vector)
        return vector if norm == 0 else vector / norm

    def model_name(self) -> str:
        return "demo-kitchen-embedder"


class DemoMergeEngine(MergeEngine):
    def __init__(self, *, settings: DriftGuardSettings):
        super().__init__(
            settings=settings,
            embedding_engine=DemoEmbeddingEngine(),
        )

    def normalize(self, text: str) -> str:
        return simple_normalize_text(text)


def build_demo_settings(
    graph_filepath: str,
    *,
    log_level: str = "WARNING",
) -> DriftGuardSettings:
    return DriftGuardSettings(
        graph_filepath=graph_filepath,
        retrieval_top_k=4,
        retrieval_min_similarity=0.58,
        traversal_max_depth=2,
        traversal_max_branching=4,
        traversal_max_paths=20,
        similarity_threshold_action=0.68,
        similarity_threshold_feedback=0.66,
        similarity_threshold_outcome=0.82,
        prune_edge_min_frequency=2,
        log_level=log_level,
    )


def phase_for_step(step_number: int) -> str:
    if step_number % 7 == 0:
        return "noise-injection"
    if step_number <= 8:
        return "seed-memory"
    if step_number <= 20:
        return "reinforce-patterns"
    return "guided-recovery"


def build_step_plan(step_number: int) -> StepPlan:
    phase = phase_for_step(step_number)

    if phase == "noise-injection":
        event = NOISE_EVENTS[((step_number // 7) - 1) % len(NOISE_EVENTS)]
        return StepPlan(
            phase=phase,
            family_name="noise",
            intent=event.action,
            risky_event=event,
            safe_action="pause and ask the lead cook for a safe fallback",
            safe_reason="noise steps exist to create weak one-off memories for pruning",
        )

    family = SCENARIO_FAMILIES[(step_number - 1) % len(SCENARIO_FAMILIES)]
    variant_index = ((step_number - 1) // len(SCENARIO_FAMILIES)) % len(
        family.risky_actions
    )

    event = RiskEvent(
        action=family.risky_actions[variant_index],
        feedback=family.feedbacks[variant_index],
        outcome=family.outcomes[variant_index],
    )

    return StepPlan(
        phase=phase,
        family_name=family.name,
        intent=event.action,
        risky_event=event,
        safe_action=family.safe_action,
        safe_reason=family.safe_reason,
    )


def should_switch_to_safe_action(
    step_number: int,
    *,
    phase: str,
    has_warning: bool,
) -> bool:
    if not has_warning:
        return False

    if phase in {"seed-memory", "noise-injection"}:
        return False

    if phase == "reinforce-patterns":
        return step_number % 4 == 0

    return step_number % 5 != 0


def should_prune(step_number: int, prune_every: int) -> bool:
    return prune_every > 0 and step_number % prune_every == 0


def summarize_event_growth(
    before: dict[str, int],
    after: dict[str, int],
    *,
    recorded: bool,
) -> dict[str, int]:
    delta_nodes = after["nodes"] - before["nodes"]
    delta_edges = after["edges"] - before["edges"]
    expected_nodes = 3 if recorded else 0
    expected_edges = 2 if recorded else 0

    return {
        "delta_nodes": delta_nodes,
        "delta_edges": delta_edges,
        "estimated_merged_nodes": max(0, expected_nodes - max(delta_nodes, 0)),
        "estimated_reused_edges": max(0, expected_edges - max(delta_edges, 0)),
    }


def build_demo_guard(
    settings: DriftGuardSettings,
    *,
    runtime_mode: str = "demo",
) -> DriftGuard:
    if runtime_mode == "real":
        return DriftGuard(settings=settings)

    runtime = build_runtime(
        settings=settings,
        merge_engine=DemoMergeEngine(settings=settings),
    )
    return DriftGuard(runtime=runtime)


def parse_args(argv: list[str] | None = None) -> Namespace:
    parser = ArgumentParser(
        description=(
            "Run a local DriftGuard demo agent that shows graph growth, "
            "merge behavior, warnings, and pruning in real time."
        )
    )
    parser.add_argument(
        "--duration-seconds",
        type=int,
        default=600,
        help="How long to run the demo loop.",
    )
    parser.add_argument(
        "--step-delay",
        type=float,
        default=5.0,
        help="Seconds to wait between steps so you can watch the output.",
    )
    parser.add_argument(
        "--prune-every",
        type=int,
        default=6,
        help="Run deep prune every N steps. Use 0 to disable scheduled pruning.",
    )
    parser.add_argument(
        "--graph-file",
        type=Path,
        default=DEFAULT_GRAPH_PATH,
        help="Where to store the demo graph JSON.",
    )
    parser.add_argument(
        "--trace-file",
        type=Path,
        default=DEFAULT_TRACE_PATH,
        help="Where to append the step-by-step demo trace as JSONL.",
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        help="DriftGuard package log level for the demo run.",
    )
    parser.add_argument(
        "--reset-graph",
        action="store_true",
        help="Delete the demo graph and trace before starting.",
    )
    parser.add_argument(
        "--runtime-mode",
        choices=("demo", "real"),
        default="demo",
        help=(
            "Use the self-contained demo runtime by default. "
            "'real' uses the package's sentence-transformers + spaCy stack."
        ),
    )
    return parser.parse_args(argv)


class KitchenLineDemoAgent:
    def __init__(
        self,
        *,
        guard: DriftGuard,
        graph_file: Path,
        trace_file: Path,
        prune_every: int,
    ):
        self.guard = guard
        self.graph_file = graph_file
        self.trace_file = trace_file
        self.prune_every = prune_every
        self.trace_file.parent.mkdir(parents=True, exist_ok=True)

    def run(self, *, duration_seconds: int, step_delay: float) -> None:
        print("DriftGuard demo task: busy kitchen line during dinner rush")
        print(f"Graph file: {self.graph_file}")
        print(f"Trace file: {self.trace_file}")
        print(
            "Goal: watch repeated kitchen mistakes become warnings, merge into "
            "shared memory, and get cleaned up by scheduled pruning.\n"
        )

        start = time.monotonic()
        step_number = 0

        while time.monotonic() - start < duration_seconds:
            step_number += 1
            self.run_step(step_number)

            if step_delay > 0 and time.monotonic() - start < duration_seconds:
                time.sleep(step_delay)

        final_stats = self.guard.stats()
        print("\nDemo finished.")
        print(
            f"Final graph stats: nodes={final_stats['nodes']} "
            f"edges={final_stats['edges']}"
        )
        print("Use the trace JSONL to inspect every step after the run.")

    def run_step(self, step_number: int) -> None:
        plan = build_step_plan(step_number)
        before_stats = self.guard.stats()
        review = self.guard.before_step(plan.intent)
        switch_to_safe = should_switch_to_safe_action(
            step_number,
            phase=plan.phase,
            has_warning=bool(review.warnings),
        )

        top_warning = review.warnings[0] if review.warnings else None

        print(
            f"=== Step {step_number:03d} | {plan.phase} | {plan.family_name} ==="
        )
        print(f"Intent: {plan.intent}")
        print(
            f"Review: warnings={len(review.warnings)} "
            f"confidence={review.confidence:.2f}"
        )

        if top_warning is not None:
            print(
                "Top warning: "
                f"trigger='{top_warning.trigger}' "
                f"risk='{top_warning.risk}' "
                f"confidence={top_warning.confidence:.2f}"
            )

        if switch_to_safe:
            after_event_stats = before_stats
            growth = summarize_event_growth(
                before_stats,
                after_event_stats,
                recorded=False,
            )
            executed_action = plan.safe_action
            recorded = False
            print(f"Decision: switched to safe action -> {executed_action}")
            print(f"Why: {plan.safe_reason}")
        else:
            self.guard.record(
                action=plan.risky_event.action,
                feedback=plan.risky_event.feedback,
                outcome=plan.risky_event.outcome,
            )
            after_event_stats = self.guard.stats()
            growth = summarize_event_growth(
                before_stats,
                after_event_stats,
                recorded=True,
            )
            executed_action = plan.risky_event.action
            recorded = True
            print(
                "Recorded mistake: "
                f"feedback='{plan.risky_event.feedback}' "
                f"outcome='{plan.risky_event.outcome}'"
            )

        print(
            "Growth: "
            f"nodes={growth['delta_nodes']:+d} "
            f"edges={growth['delta_edges']:+d} "
            f"merged_nodes~{growth['estimated_merged_nodes']} "
            f"reused_edges~{growth['estimated_reused_edges']}"
        )

        prune_result = None
        if should_prune(step_number, self.prune_every):
            prune_result = self.guard.prune()
            before_prune = prune_result["before"]
            after_prune = prune_result["after"]
            print(
                "Prune: "
                f"nodes {before_prune['nodes']} -> {after_prune['nodes']}, "
                f"edges {before_prune['edges']} -> {after_prune['edges']}"
            )
        else:
            print("Prune: skipped this step")

        final_stats = self.guard.stats()
        print(
            f"Graph stats now: nodes={final_stats['nodes']} "
            f"edges={final_stats['edges']}\n"
        )

        self._append_trace(
            {
                "step": step_number,
                "phase": plan.phase,
                "family": plan.family_name,
                "intent": plan.intent,
                "executed_action": executed_action,
                "recorded_mistake": recorded,
                "warnings_count": len(review.warnings),
                "review_confidence": round(review.confidence, 4),
                "top_warning": (
                    {
                        "trigger": top_warning.trigger,
                        "risk": top_warning.risk,
                        "confidence": round(top_warning.confidence, 4),
                    }
                    if top_warning is not None
                    else None
                ),
                "risky_event": asdict(plan.risky_event),
                "safe_action": plan.safe_action,
                "growth": growth,
                "stats_before": before_stats,
                "stats_after_event": after_event_stats,
                "stats_final": final_stats,
                "prune": prune_result,
            }
        )

    def _append_trace(self, payload: dict) -> None:
        with open(self.trace_file, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")


def _reset_demo_files(graph_file: Path, trace_file: Path) -> None:
    for path in (graph_file, trace_file):
        if path.exists():
            path.unlink()


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    configure_logging(args.log_level)
    settings = build_demo_settings(
        str(args.graph_file),
        log_level=args.log_level,
    )

    if args.reset_graph:
        _reset_demo_files(args.graph_file, args.trace_file)

    agent = KitchenLineDemoAgent(
        guard=build_demo_guard(settings, runtime_mode=args.runtime_mode),
        graph_file=args.graph_file,
        trace_file=args.trace_file,
        prune_every=args.prune_every,
    )

    try:
        agent.run(
            duration_seconds=args.duration_seconds,
            step_delay=args.step_delay,
        )
    except KeyboardInterrupt:
        final_stats = agent.guard.stats()
        print("\nDemo interrupted by user.")
        print(
            f"Graph preserved at {args.graph_file} with "
            f"nodes={final_stats['nodes']} edges={final_stats['edges']}"
        )


if __name__ == "__main__":
    main()
