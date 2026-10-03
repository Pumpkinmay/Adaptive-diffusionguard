"""Provider-neutral managed LLM integration."""

from .action_gateway import (
    ActionDecisionError,
    ActionDecisionGateway,
    GatewayAudit,
    attach_action_gateway,
)
from .model_factory import LLMSettings, create_llm_model
from .runtime import CallLimitExceeded, LLMRuntime, ManagedModelBackend
from .structured_actions import (
    ActionMask,
    ActionMaskBuilder,
    DecisionSnapshot,
    StructuredActionResponse,
    build_decision_messages,
    build_response_format,
    parse_structured_response,
    revalidate_snapshot_choice,
)
from .structured_retry import (
    CORRECTION_PROMPT,
    StructuredAttemptResult,
    StructuredRequest,
    request_with_structured_correction,
)

__all__ = [
    "CORRECTION_PROMPT",
    "ActionDecisionError",
    "ActionDecisionGateway",
    "ActionMask",
    "ActionMaskBuilder",
    "CallLimitExceeded",
    "DecisionSnapshot",
    "GatewayAudit",
    "LLMRuntime",
    "LLMSettings",
    "ManagedModelBackend",
    "StructuredActionResponse",
    "StructuredAttemptResult",
    "StructuredRequest",
    "attach_action_gateway",
    "build_decision_messages",
    "build_response_format",
    "create_llm_model",
    "parse_structured_response",
    "request_with_structured_correction",
    "revalidate_snapshot_choice",
]
