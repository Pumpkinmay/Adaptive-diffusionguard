"""Teacher behavior distillation with strict output normalization."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import replace

from camel.models import BaseModelBackend

from adaptive_diffusionguard.llm.actions import (
    ActionValidationMetrics,
    extract_response_text,
    normalize_action,
)
from training.schemas import BehaviorExample

SYSTEM_PROMPT = (
    "You are a synthetic social-media behavior teacher. Return only JSON with "
    "exact keys action, confidence, reason. action must be one of repost, quote, "
    "report, ignore. confidence must be between 0 and 1."
)

_SECRET_PATTERN = re.compile(r"(?:gsk_[A-Za-z0-9_-]{12,}|sk-[A-Za-z0-9_-]{12,})")
_EMAIL_PATTERN = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
_PHONE_PATTERN = re.compile(r"(?<!\d)(?:\+?\d[\d ()-]{7,}\d)(?!\d)")


def redact_sensitive(value: str) -> str:
    value = _SECRET_PATTERN.sub("[REDACTED_SECRET]", value)
    value = _EMAIL_PATTERN.sub("[REDACTED_EMAIL]", value)
    return _PHONE_PATTERN.sub("[REDACTED_PHONE]", value)


class TeacherTrajectoryGenerator:
    """Generate synthetic teacher labels; never presents them as human labels."""

    def __init__(self, model: BaseModelBackend) -> None:
        self.model = model
        self.metrics = ActionValidationMetrics()

    def _complete(self, system_prompt: str, user_prompt: str) -> str:
        response = self.model.run(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
        return extract_response_text(response)

    def generate_one(self, example: BehaviorExample) -> BehaviorExample:
        raw = self._complete(SYSTEM_PROMPT, redact_sensitive(example.prompt()))

        def repair(invalid: str) -> str:
            prompt = (
                "Repair the following output to strict JSON only. Do not add "
                "new facts. Allowed actions: repost, quote, report, ignore.\n"
                + redact_sensitive(invalid)
            )
            return self._complete(SYSTEM_PROMPT, prompt)

        result = normalize_action(raw, repair=repair)
        self.metrics.observe(result)
        safe_label = result.output.model_copy(
            update={"reason": redact_sensitive(result.output.reason)}
        )
        return replace(
            example,
            user_profile=redact_sensitive(example.user_profile),
            feed_post=redact_sensitive(example.feed_post),
            neighbor_interactions=[redact_sensitive(v) for v in example.neighbor_interactions],
            behavior_history=[redact_sensitive(v) for v in example.behavior_history],
            platform_notice=redact_sensitive(example.platform_notice),
            label=safe_label,
            label_source="teacher_synthetic",
        )

    def generate(self, examples: Iterable[BehaviorExample]) -> list[BehaviorExample]:
        return [self.generate_one(example) for example in examples]
