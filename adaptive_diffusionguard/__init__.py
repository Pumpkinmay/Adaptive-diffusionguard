"""Adaptive DiffusionGuard public API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from adaptive_diffusionguard.governance.controller import RuleBasedController
from adaptive_diffusionguard.governance.cosref import StaticCOSREFPolicy

if TYPE_CHECKING:
    from adaptive_diffusionguard.platform import AdaptiveDiffusionPlatform

__all__ = [
    "AdaptiveDiffusionPlatform",
    "RuleBasedController",
    "StaticCOSREFPolicy",
]


def __getattr__(name: str) -> Any:
    """Avoid importing OASIS and creating its log file for non-platform tools."""
    if name == "AdaptiveDiffusionPlatform":
        from adaptive_diffusionguard.platform import AdaptiveDiffusionPlatform

        return AdaptiveDiffusionPlatform
    raise AttributeError(name)
