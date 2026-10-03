"""Strict schemas for distilled or observed action examples."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal

from adaptive_diffusionguard.llm.actions import VALID_ACTIONS, ActionOutput

Action = Literal["repost", "quote", "report", "ignore"]
LabelSource = Literal["observed", "teacher_synthetic"]
Provenance = Literal[
    "decision_snapshot",
    "legacy_refresh_trace_reconstruction",
    "demo",
    "unversioned",
]
VALID_LABEL_SOURCES = frozenset({"observed", "teacher_synthetic"})


@dataclass(frozen=True, slots=True)
class BehaviorExample:
    sample_id: str
    user_profile: str
    community: str
    feed_post: str
    neighbor_interactions: list[str]
    behavior_history: list[str]
    platform_notice: str
    label: ActionOutput
    label_source: LabelSource
    action_trace_rowid: int | None = None
    decision_sequence: int | None = None
    decision_timestep: int | None = None
    visible_post_ids: list[int] = field(default_factory=list)
    legal_choice_ids: list[str] = field(default_factory=list)
    selected_choice_id: str | None = None
    provenance: Provenance = "unversioned"
    action_label_eligible: bool = True
    rationale_training_eligible: bool = True
    rationale_quality_status: str = "unreviewed"
    feed_empty_reason: str | None = None

    def __post_init__(self) -> None:
        if self.label_source not in VALID_LABEL_SOURCES:
            raise ValueError(f"invalid label_source: {self.label_source}")

    def prompt(self) -> str:
        payload = {
            "user_profile": self.user_profile,
            "community": self.community,
            "feed_post": self.feed_post,
            "legal_choice_ids": self.legal_choice_ids,
            "neighbor_interactions": self.neighbor_interactions,
            "behavior_history": self.behavior_history,
            "platform_notice": self.platform_notice,
        }
        return (
            "Choose one action and return only strict JSON with keys action, "
            "confidence, reason. Allowed actions: repost, quote, report, ignore.\n"
            + json.dumps(payload, ensure_ascii=False, sort_keys=True)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "action_trace_rowid": self.action_trace_rowid,
            "decision_sequence": self.decision_sequence,
            "decision_timestep": self.decision_timestep,
            "user_profile": self.user_profile,
            "community": self.community,
            "feed_post": self.feed_post,
            "neighbor_interactions": self.neighbor_interactions,
            "behavior_history": self.behavior_history,
            "platform_notice": self.platform_notice,
            "label": self.label.model_dump(),
            "label_source": self.label_source,
            "visible_post_ids": self.visible_post_ids,
            "legal_choice_ids": self.legal_choice_ids,
            "selected_choice_id": self.selected_choice_id,
            "provenance": self.provenance,
            "action_label_eligible": self.action_label_eligible,
            "rationale_training_eligible": self.rationale_training_eligible,
            "rationale_quality_status": self.rationale_quality_status,
            "feed_empty_reason": self.feed_empty_reason,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> BehaviorExample:
        data = dict(payload)
        data["label"] = ActionOutput.model_validate(data["label"])
        return cls(**data)


__all__ = [
    "VALID_ACTIONS",
    "VALID_LABEL_SOURCES",
    "Action",
    "ActionOutput",
    "BehaviorExample",
    "LabelSource",
    "Provenance",
]
