"""Single provider-aware model construction entry point."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from camel.models import ModelFactory
from camel.types import ModelPlatformType

from .runtime import LLMRuntime, ManagedModelBackend, ResponseCache


class LLMConfigurationError(ValueError):
    """Raised before any remote request when LLM settings are invalid."""


def _parse_bool(name: str, value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise LLMConfigurationError(f"{name} must be true or false")


def _positive_int(name: str, value: str, *, allow_zero: bool = False) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise LLMConfigurationError(f"{name} must be an integer") from exc
    lower = 0 if allow_zero else 1
    if parsed < lower:
        raise LLMConfigurationError(f"{name} must be >= {lower}")
    return parsed


def _positive_float(name: str, value: str, *, allow_zero: bool = False) -> float:
    try:
        parsed = float(value)
    except ValueError as exc:
        raise LLMConfigurationError(f"{name} must be numeric") from exc
    if parsed < 0 or (parsed == 0 and not allow_zero):
        comparator = ">= 0" if allow_zero else "> 0"
        raise LLMConfigurationError(f"{name} must be {comparator}")
    return parsed


@dataclass(frozen=True, slots=True)
class LLMSettings:
    """Validated LLM configuration. Secret fields are excluded from repr."""

    enabled: bool = False
    provider: str = "groq"
    llm_model: str = "openai/gpt-oss-20b"
    groq_api_key: str | None = field(default=None, repr=False)
    openai_api_key: str | None = field(default=None, repr=False)
    temperature: float = 0.2
    max_tokens: int = 256
    timeout_seconds: float = 60.0
    max_retries: int = 1
    max_concurrency: int = 1
    max_calls_per_run: int = 30
    cache_enabled: bool = True
    request_interval_seconds: float = 8.0
    cache_path: Path = Path(".cache/llm_responses.sqlite3")

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> LLMSettings:
        source = os.environ if env is None else env
        settings = cls(
            enabled=_parse_bool(
                "DIFFUSIONGUARD_ENABLE_LLM",
                source.get("DIFFUSIONGUARD_ENABLE_LLM", "false"),
            ),
            provider=source.get("LLM_PROVIDER", "groq").strip().lower(),
            llm_model=source.get("LLM_MODEL", "openai/gpt-oss-20b").strip(),
            groq_api_key=source.get("GROQ_API_KEY") or None,
            openai_api_key=source.get("OPENAI_API_KEY") or None,
            temperature=_positive_float(
                "LLM_TEMPERATURE",
                source.get("LLM_TEMPERATURE", "0.2"),
                allow_zero=True,
            ),
            max_tokens=_positive_int(
                "LLM_MAX_TOKENS", source.get("LLM_MAX_TOKENS", "256")
            ),
            timeout_seconds=_positive_float(
                "LLM_TIMEOUT_SECONDS",
                source.get("LLM_TIMEOUT_SECONDS", "60"),
            ),
            max_retries=_positive_int(
                "LLM_MAX_RETRIES",
                source.get("LLM_MAX_RETRIES", "1"),
                allow_zero=True,
            ),
            max_concurrency=_positive_int(
                "LLM_MAX_CONCURRENCY",
                source.get("LLM_MAX_CONCURRENCY", "1"),
            ),
            max_calls_per_run=_positive_int(
                "LLM_MAX_CALLS_PER_RUN",
                source.get("LLM_MAX_CALLS_PER_RUN", "30"),
            ),
            cache_enabled=_parse_bool(
                "LLM_CACHE_ENABLED", source.get("LLM_CACHE_ENABLED", "true")
            ),
            request_interval_seconds=_positive_float(
                "LLM_REQUEST_INTERVAL_SECONDS",
                source.get("LLM_REQUEST_INTERVAL_SECONDS", "8"),
                allow_zero=True,
            ),
            cache_path=Path(
                source.get("LLM_CACHE_PATH", ".cache/llm_responses.sqlite3")
            ),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        if self.provider not in {"groq", "openai"}:
            raise LLMConfigurationError(
                f"unsupported LLM_PROVIDER {self.provider!r}; use groq or openai"
            )
        if not self.llm_model:
            raise LLMConfigurationError("LLM_MODEL cannot be empty")
        if not 0.0 <= self.temperature <= 2.0:
            raise LLMConfigurationError("LLM_TEMPERATURE must be in [0, 2]")
        if self.enabled and self.provider == "groq" and not self.groq_api_key:
            raise LLMConfigurationError(
                "GROQ_API_KEY is required when LLM_PROVIDER=groq"
            )
        if self.enabled and self.provider == "openai" and not self.openai_api_key:
            raise LLMConfigurationError(
                "OPENAI_API_KEY is required when LLM_PROVIDER=openai"
            )


def create_llm_model(
    settings: LLMSettings,
    *,
    factory: Callable[..., Any] = ModelFactory.create,
    sleeper: Callable[[float], None] | None = None,
) -> ManagedModelBackend | None:
    """Create one managed CAMEL backend, or no backend when disabled."""
    settings.validate()
    if not settings.enabled:
        return None

    if settings.provider == "groq":
        platform = ModelPlatformType.GROQ
        api_key = settings.groq_api_key
    else:
        platform = ModelPlatformType.OPENAI
        api_key = settings.openai_api_key

    model_config_dict = {
        "temperature": settings.temperature,
        "max_tokens": settings.max_tokens,
    }
    backend = factory(
        model_platform=platform,
        model_type=settings.llm_model,
        api_key=api_key,
        model_config_dict=model_config_dict,
        timeout=settings.timeout_seconds,
        # The managed outer layer owns retries so the hard call cap is exact.
        max_retries=0,
    )
    cache = ResponseCache(settings.cache_path) if settings.cache_enabled else None
    runtime = LLMRuntime(
        settings,
        cache=cache,
        model_config=getattr(backend, "model_config_dict", model_config_dict),
        sleeper=sleeper,
    )
    return ManagedModelBackend(backend, runtime)
