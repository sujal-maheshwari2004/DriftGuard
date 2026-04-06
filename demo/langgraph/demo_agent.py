from __future__ import annotations

from argparse import ArgumentParser, Namespace
from dataclasses import asdict, dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, TypedDict
import json
import os
import sys
import time


ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from driftguard import DriftGuard, DriftGuardSettings, configure_logging
from demo.rule_based.demo_agent import (
    SCENARIO_FAMILIES,
    build_demo_guard,
    build_demo_settings,
    should_prune,
    simple_normalize_text,
    summarize_event_growth,
)


DEFAULT_GRAPH_PATH = (
    ROOT_DIR / "demo" / "output" / "langgraph" / "kitchen_line_graph.json"
)
DEFAULT_TRACE_PATH = (
    ROOT_DIR / "demo" / "output" / "langgraph" / "kitchen_line_trace.jsonl"
)

SYSTEM_PROMPT = """
You are an autonomous kitchen line agent in a dinner-rush simulation.
Produce exactly one concrete next action at a time.
Keep actions short, imperative, and physically plausible in a real kitchen.
When DriftGuard warns that a similar action caused problems before, revise toward a safer alternative.
Return strict JSON with keys:
- thought: short reasoning
- action: the exact next action
""".strip()


class LangGraphDemoDependencyError(RuntimeError):
    """Raised when the LangGraph demo dependencies are unavailable."""


@dataclass(frozen=True)
class ExecutionAssessment:
    family_name: str
    recorded_mistake: bool
    action: str
    feedback: str | None
    outcome: str | None
    observation: str


class DemoState(TypedDict, total=False):
    task: str
    step: int
    started_at: float
    duration_seconds: int
    step_delay: float
    prune_every: int
    trace_file: str
    candidate_action: str
    final_action: str
    planner_thought: str
    revision_thought: str
    review_confidence: float
    warnings_count: int
    top_warning: dict[str, Any] | None
    growth: dict[str, int]
    stats_before: dict[str, int]
    stats_after: dict[str, int]
    last_observation: str
    prune: dict[str, Any] | None


FAMILY_BY_NAME = {family.name: family for family in SCENARIO_FAMILIES}
FAMILY_MATCHERS = {
    "seasoning": {"salt", "taste"},
    "heat-control": {"heat", "burn"},
    "oil-management": {"oil", "grease"},
    "resting": {"rest", "plate", "dry"},
}
SAFE_TOKENS = {
    "seasoning": {"taste"},
    "heat-control": {"lower", "extend", "gentle"},
    "oil-management": {"measure", "small"},
    "resting": {"rest"},
}


def build_langgraph_settings(
    graph_filepath: str,
    *,
    log_level: str = "WARNING",
) -> DriftGuardSettings:
    return build_demo_settings(graph_filepath, log_level=log_level)


def parse_llm_json_response(text: str) -> dict[str, str]:
    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end > start:
        try:
            payload = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            payload = {}
    else:
        payload = {}

    action = str(payload.get("action") or payload.get("next_action") or text).strip()
    thought = str(payload.get("thought") or payload.get("reasoning") or "").strip()
    return {
        "thought": thought,
        "action": action,
    }


def family_name_for_action(action: str) -> str:
    normalized = simple_normalize_text(action)
    tokens = set(normalized.split())

    for family_name, family_tokens in FAMILY_MATCHERS.items():
        if tokens & family_tokens:
            return family_name

    return "unknown"


def should_revise_action(
    *,
    warnings_count: int,
    review_confidence: float,
    threshold: float = 0.72,
) -> bool:
    return warnings_count > 0 and review_confidence >= threshold


def assess_kitchen_action(action: str, step_number: int) -> ExecutionAssessment:
    family_name = family_name_for_action(action)

    if family_name == "unknown":
        return ExecutionAssessment(
            family_name=family_name,
            recorded_mistake=False,
            action=action,
            feedback=None,
            outcome=None,
            observation="Action did not map to a known risky kitchen family.",
        )

    normalized = simple_normalize_text(action)
    tokens = set(normalized.split())
    family = FAMILY_BY_NAME[family_name]

    if tokens & SAFE_TOKENS[family_name]:
        return ExecutionAssessment(
            family_name=family_name,
            recorded_mistake=False,
            action=action,
            feedback=None,
            outcome=None,
            observation=family.safe_reason,
        )

    variant = (step_number - 1) % len(family.feedbacks)
    return ExecutionAssessment(
        family_name=family_name,
        recorded_mistake=True,
        action=action,
        feedback=family.feedbacks[variant],
        outcome=family.outcomes[variant],
        observation=f"Risky {family_name} action reinforced an existing failure mode.",
    )


def _response_text(message: Any) -> str:
    content = getattr(message, "content", message)

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(str(item.get("text", "")))
        return "\n".join(part for part in parts if part)

    return str(content)


def _load_langgraph_dependencies():
    try:
        graph_module = import_module("langgraph.graph")
        openai_module = import_module("langchain_openai")
    except Exception as exc:
        raise LangGraphDemoDependencyError(
            "The LangGraph demo needs extra dependencies. "
            "Install them with: pip install -e \".[demo]\""
        ) from exc

    return (
        graph_module.END,
        graph_module.StateGraph,
        openai_module.ChatOpenAI,
    )


def _build_model(*, model_name: str, temperature: float):
    _, _, chat_open_ai = _load_langgraph_dependencies()
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise LangGraphDemoDependencyError(
            "OPENAI_API_KEY is required for the LangGraph demo."
        )

    kwargs: dict[str, Any] = {
        "model": model_name,
        "temperature": temperature,
        "api_key": api_key,
    }

    if base_url := os.getenv("OPENAI_BASE_URL"):
        kwargs["base_url"] = base_url

    return chat_open_ai(**kwargs)


def _reset_demo_files(graph_file: Path, trace_file: Path) -> None:
    for path in (graph_file, trace_file):
        if path.exists():
            path.unlink()


def _append_trace(trace_file: Path, payload: dict[str, Any]) -> None:
    trace_file.parent.mkdir(parents=True, exist_ok=True)
    with open(trace_file, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")


def _planner_prompt(state: DemoState) -> str:
    warning_line = "No warnings yet."

    if state.get("top_warning"):
        warning = state["top_warning"]
        warning_line = (
            f"Top warning: trigger={warning['trigger']!r}, "
            f"risk={warning['risk']!r}, confidence={warning['confidence']:.2f}"
        )

    return (
        f"Task: {state['task']}\n"
        f"Step: {state['step']}\n"
        f"Last observation: {state.get('last_observation', 'Start service.')}\n"
        f"{warning_line}\n"
        "Propose the next single action as JSON."
    )


def _revision_prompt(state: DemoState) -> str:
    warning = state.get("top_warning") or {}
    return (
        f"Original action: {state['candidate_action']}\n"
        f"DriftGuard warning count: {state['warnings_count']}\n"
        f"Top warning trigger: {warning.get('trigger', 'n/a')}\n"
        f"Top warning risk: {warning.get('risk', 'n/a')}\n"
        f"Top warning confidence: {warning.get('confidence', 0.0):.2f}\n"
        "Revise to a safer next action and return strict JSON."
    )


def parse_args(argv: list[str] | None = None) -> Namespace:
    parser = ArgumentParser(
        description="Run a LangGraph-based DriftGuard demo agent."
    )
    parser.add_argument(
        "--duration-seconds",
        type=int,
        default=120,
        help="How long to run the demo loop.",
    )
    parser.add_argument(
        "--step-delay",
        type=float,
        default=4.0,
        help="Seconds to wait between steps so you can watch the output.",
    )
    parser.add_argument(
        "--prune-every",
        type=int,
        default=4,
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
        "--runtime-mode",
        choices=("demo", "real"),
        default="demo",
        help=(
            "Use the self-contained DriftGuard demo runtime by default. "
            "'real' uses the package's sentence-transformers + spaCy stack."
        ),
    )
    parser.add_argument(
        "--model",
        default=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        help="OpenAI-compatible chat model name.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="LLM temperature used for planning and revision.",
    )
    parser.add_argument(
        "--reset-graph",
        action="store_true",
        help="Delete the demo graph and trace before starting.",
    )
    return parser.parse_args(argv)


def build_langgraph_app(
    *,
    guard: DriftGuard,
    model: Any,
    trace_file: Path,
    prune_every: int,
):
    end, state_graph_cls, _ = _load_langgraph_dependencies()
    graph = state_graph_cls(DemoState)

    def draft_action(state: DemoState) -> DemoState:
        step = state.get("step", 0) + 1
        response = model.invoke(
            [
                ("system", SYSTEM_PROMPT),
                ("human", _planner_prompt({**state, "step": step})),
            ]
        )
        parsed = parse_llm_json_response(_response_text(response))
        return {
            "step": step,
            "candidate_action": parsed["action"],
            "final_action": parsed["action"],
            "planner_thought": parsed["thought"],
            "stats_before": guard.stats(),
        }

    def review_action(state: DemoState) -> DemoState:
        review = guard.before_step(state["candidate_action"])
        top_warning = review.warnings[0] if review.warnings else None
        return {
            "warnings_count": len(review.warnings),
            "review_confidence": review.confidence,
            "top_warning": (
                {
                    "trigger": top_warning.trigger,
                    "risk": top_warning.risk,
                    "confidence": top_warning.confidence,
                }
                if top_warning is not None
                else None
            ),
        }

    def revise_action(state: DemoState) -> DemoState:
        response = model.invoke(
            [
                ("system", SYSTEM_PROMPT),
                ("human", _revision_prompt(state)),
            ]
        )
        parsed = parse_llm_json_response(_response_text(response))
        return {
            "final_action": parsed["action"],
            "revision_thought": parsed["thought"],
        }

    def execute_action(state: DemoState) -> DemoState:
        action = state.get("final_action", state["candidate_action"])
        before_stats = state["stats_before"]
        assessment = assess_kitchen_action(action, state["step"])

        if assessment.recorded_mistake:
            guard.record(
                action=assessment.action,
                feedback=assessment.feedback or "",
                outcome=assessment.outcome or "",
            )

        after_stats = guard.stats()
        growth = summarize_event_growth(
            before_stats,
            after_stats,
            recorded=assessment.recorded_mistake,
        )

        prune_result = None
        if should_prune(state["step"], prune_every):
            prune_result = guard.prune()

        final_stats = guard.stats()
        payload = {
            "step": state["step"],
            "candidate_action": state["candidate_action"],
            "final_action": action,
            "planner_thought": state.get("planner_thought", ""),
            "revision_thought": state.get("revision_thought", ""),
            "warnings_count": state.get("warnings_count", 0),
            "review_confidence": round(state.get("review_confidence", 0.0), 4),
            "top_warning": state.get("top_warning"),
            "assessment": asdict(assessment),
            "growth": growth,
            "stats_before": before_stats,
            "stats_after": final_stats,
            "prune": prune_result,
        }
        _append_trace(trace_file, payload)

        print(f"=== LangGraph Step {state['step']:03d} ===")
        print(f"Draft action: {state['candidate_action']}")
        if state.get("top_warning") is not None:
            warning = state["top_warning"]
            print(
                "DriftGuard warning: "
                f"trigger='{warning['trigger']}' "
                f"risk='{warning['risk']}' "
                f"confidence={warning['confidence']:.2f}"
            )
        else:
            print("DriftGuard warning: none")
        print(f"Final action: {action}")
        print(f"Observation: {assessment.observation}")
        print(
            f"Growth: nodes={growth['delta_nodes']:+d} "
            f"edges={growth['delta_edges']:+d} "
            f"merged_nodes~{growth['estimated_merged_nodes']}"
        )
        if prune_result is not None:
            print(
                "Prune: "
                f"nodes {prune_result['before']['nodes']} -> {prune_result['after']['nodes']}, "
                f"edges {prune_result['before']['edges']} -> {prune_result['after']['edges']}"
            )
        else:
            print("Prune: skipped this step")
        print(
            f"Graph stats now: nodes={final_stats['nodes']} "
            f"edges={final_stats['edges']}\n"
        )

        return {
            "growth": growth,
            "stats_after": after_stats,
            "prune": prune_result,
            "last_observation": assessment.observation,
        }

    def route_after_review(state: DemoState) -> str:
        if should_revise_action(
            warnings_count=state.get("warnings_count", 0),
            review_confidence=state.get("review_confidence", 0.0),
        ):
            return "revise_action"
        return "execute_action"

    graph.add_node("draft_action", draft_action)
    graph.add_node("review_action", review_action)
    graph.add_node("revise_action", revise_action)
    graph.add_node("execute_action", execute_action)

    graph.set_entry_point("draft_action")
    graph.add_edge("draft_action", "review_action")
    graph.add_conditional_edges(
        "review_action",
        route_after_review,
        {
            "revise_action": "revise_action",
            "execute_action": "execute_action",
        },
    )
    graph.add_edge("revise_action", "execute_action")
    graph.add_edge("execute_action", end)

    return graph.compile()


def run_langgraph_demo(
    *,
    task: str,
    duration_seconds: int,
    step_delay: float,
    prune_every: int,
    graph_file: Path,
    trace_file: Path,
    log_level: str,
    runtime_mode: str,
    model_name: str,
    temperature: float,
    reset_graph: bool,
) -> None:
    configure_logging(log_level)
    settings = build_langgraph_settings(str(graph_file), log_level=log_level)

    if reset_graph:
        _reset_demo_files(graph_file, trace_file)

    guard = build_demo_guard(settings, runtime_mode=runtime_mode)
    model = _build_model(model_name=model_name, temperature=temperature)
    app = build_langgraph_app(
        guard=guard,
        model=model,
        trace_file=trace_file,
        prune_every=prune_every,
    )

    print("DriftGuard LangGraph demo task: busy kitchen line during dinner rush")
    print(f"Graph file: {graph_file}")
    print(f"Trace file: {trace_file}")
    print(f"Model: {model_name}")
    print(
        "Goal: watch an LLM propose actions, DriftGuard review them, and "
        "the graph evolve over a 120-second run.\n"
    )

    started_at = time.monotonic()
    state: DemoState = {
        "task": task,
        "step": 0,
        "started_at": started_at,
        "duration_seconds": duration_seconds,
        "step_delay": step_delay,
        "prune_every": prune_every,
        "trace_file": str(trace_file),
        "last_observation": "Service has just started.",
    }

    while time.monotonic() - started_at < duration_seconds:
        state = app.invoke(state)

        if step_delay > 0 and time.monotonic() - started_at < duration_seconds:
            time.sleep(step_delay)

    final_stats = guard.stats()
    print("\nLangGraph demo finished.")
    print(
        f"Final graph stats: nodes={final_stats['nodes']} "
        f"edges={final_stats['edges']}"
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    try:
        run_langgraph_demo(
            task=(
                "Manage a busy kitchen line during dinner rush. "
                "Keep orders moving and improve from mistakes."
            ),
            duration_seconds=args.duration_seconds,
            step_delay=args.step_delay,
            prune_every=args.prune_every,
            graph_file=args.graph_file,
            trace_file=args.trace_file,
            log_level=args.log_level,
            runtime_mode=args.runtime_mode,
            model_name=args.model,
            temperature=args.temperature,
            reset_graph=args.reset_graph,
        )
    except KeyboardInterrupt:
        print("\nLangGraph demo interrupted by user.")


if __name__ == "__main__":
    main()
