from functools import wraps
from typing import Any, Callable, Literal

from driftguard.config import DEFAULT_SETTINGS, DriftGuardSettings
from driftguard.logging_config import get_logger
from driftguard.models.response import RetrievalResponse
from driftguard.runtime import DriftGuardRuntime, build_runtime


logger = get_logger(__name__)
GuardPolicy = Literal["warn", "block", "acknowledge", "record_only"]


class GuardrailTriggered(RuntimeError):
    """
    Raised when DriftGuard detects a risky repeated action and blocking is enabled.
    """


class GuardrailAcknowledgementRequired(RuntimeError):
    """
    Raised when DriftGuard requires explicit acknowledgement before continuing.
    """


class DriftGuard:
    """
    In-process guardrail API for agent frameworks and step hooks.
    """

    def __init__(
        self,
        runtime: DriftGuardRuntime | None = None,
        *,
        settings: DriftGuardSettings | None = None,
    ):
        self.runtime = runtime or build_runtime(settings=settings)
        self.settings = settings or getattr(self.runtime, "settings", DEFAULT_SETTINGS)

    def review(self, context: str):
        return self.runtime.query_memory(context)

    def record(
        self,
        action: str,
        feedback: str,
        outcome: str,
    ) -> dict:
        return self.runtime.register_mistake(
            action=action,
            feedback=feedback,
            outcome=outcome,
        )

    def prune(self) -> dict:
        return self.runtime.deep_prune()

    def stats(self) -> dict:
        return self.runtime.graph_stats()

    def before_step(
        self,
        context: str,
        *,
        min_confidence: float | None = None,
        raise_on_match: bool = False,
        policy: GuardPolicy | None = None,
        acknowledged: bool = False,
    ):
        resolved_policy = self._resolve_policy(
            policy=policy,
            raise_on_match=raise_on_match,
        )
        min_confidence = self._resolve_min_confidence(min_confidence)

        if resolved_policy == "record_only":
            logger.info(
                "Skipping pre-step review for context=%r due to record_only policy",
                context,
            )
            self.runtime.metrics.record_review(
                warnings_count=0,
                confidence=0.0,
                skipped=True,
            )
            return RetrievalResponse(
                query=context,
                warnings=[],
                chains=[],
                confidence=0.0,
            )

        response = self.review(context)

        if self._should_block(
            response,
            min_confidence=min_confidence,
            policy=resolved_policy,
        ):
            logger.warning(
                "Blocking agent step for context=%r with confidence=%.2f",
                context,
                response.confidence,
            )
            self.runtime.metrics.record_review(
                warnings_count=len(response.warnings),
                confidence=response.confidence,
                blocked=True,
            )
            raise GuardrailTriggered(self._format_block_message(context, response))

        if self._should_acknowledge(
            response,
            min_confidence=min_confidence,
            policy=resolved_policy,
            acknowledged=acknowledged,
        ):
            logger.warning(
                "Acknowledgement required for context=%r with confidence=%.2f",
                context,
                response.confidence,
            )
            self.runtime.metrics.record_review(
                warnings_count=len(response.warnings),
                confidence=response.confidence,
                acknowledgement_required=True,
            )
            raise GuardrailAcknowledgementRequired(
                self._format_acknowledgement_message(context, response)
            )

        return response

    def _should_block(
        self,
        response,
        *,
        min_confidence: float,
        policy: GuardPolicy,
    ) -> bool:
        return (
            policy == "block"
            and bool(response.warnings)
            and response.confidence >= min_confidence
        )

    def _should_acknowledge(
        self,
        response,
        *,
        min_confidence: float,
        policy: GuardPolicy,
        acknowledged: bool,
    ) -> bool:
        return (
            policy == "acknowledge"
            and not acknowledged
            and bool(response.warnings)
            and response.confidence >= min_confidence
        )

    def _resolve_policy(
        self,
        *,
        policy: GuardPolicy | None,
        raise_on_match: bool,
    ) -> GuardPolicy:
        if raise_on_match and policy in (None, "warn"):
            return "block"

        return policy or self.settings.guard_policy

    def _resolve_min_confidence(self, min_confidence: float | None) -> float:
        if min_confidence is None:
            return self.settings.guard_min_confidence
        return min_confidence

    def _format_block_message(self, context: str, response) -> str:
        top_warning = response.warnings[0] if response.warnings else None

        if top_warning is None:
            return f"DriftGuard blocked the step for context={context!r}"

        return (
            f"DriftGuard blocked the step for context={context!r}. "
            f"Top warning: trigger={top_warning.trigger!r}, risk={top_warning.risk!r}, "
            f"confidence={top_warning.confidence:.2f}"
        )

    def _format_acknowledgement_message(self, context: str, response) -> str:
        top_warning = response.warnings[0] if response.warnings else None

        if top_warning is None:
            return (
                f"DriftGuard requires acknowledgement before continuing "
                f"for context={context!r}"
            )

        return (
            f"DriftGuard requires acknowledgement for context={context!r}. "
            f"Top warning: trigger={top_warning.trigger!r}, risk={top_warning.risk!r}, "
            f"confidence={top_warning.confidence:.2f}"
        )


def guard_step(
    guard: DriftGuard,
    *,
    input_getter: Callable[..., str] | None = None,
    min_confidence: float | None = None,
    raise_on_match: bool = False,
    policy: GuardPolicy | None = None,
    acknowledged_getter: Callable[..., bool] | None = None,
    on_review: Callable[[Any], None] | None = None,
):
    """
    Decorate an agent step so DriftGuard reviews the intended action beforehand.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            context = (
                input_getter(*args, **kwargs)
                if input_getter is not None
                else _default_context(*args, **kwargs)
            )

            logger.info(
                "Running guard review before agent step function=%s context=%r",
                func.__name__,
                context,
            )
            review = guard.before_step(
                context,
                min_confidence=min_confidence,
                raise_on_match=raise_on_match,
                policy=policy,
                acknowledged=(
                    acknowledged_getter(*args, **kwargs)
                    if acknowledged_getter is not None
                    else False
                ),
            )

            if on_review is not None:
                on_review(review)

            return func(*args, **kwargs)

        return wrapper

    return decorator


def _default_context(*args, **kwargs) -> str:
    if args and isinstance(args[0], str):
        return args[0]

    for key in ("context", "task", "prompt", "action"):
        value = kwargs.get(key)
        if isinstance(value, str):
            return value

    raise ValueError(
        "Unable to derive agent step context automatically. "
        "Pass a string first argument or provide input_getter=..."
    )
