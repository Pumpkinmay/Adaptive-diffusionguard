"""Rate, retry, budget, cache, and usage controls for CAMEL backends."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import re
import sqlite3
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from camel.models import BaseModelBackend
from camel.types import ChatCompletion

if TYPE_CHECKING:
    from .model_factory import LLMSettings

CACHE_SCHEMA_VERSION = "diffusionguard-llm-cache-v2"
_SENSITIVE_CONFIG_KEYS = {
    "api_key",
    "apikey",
    "access_token",
    "auth_token",
    "authorization",
    "secret",
}


def _is_sensitive_config_key(key: Any) -> bool:
    normalized = str(key).lower()
    return normalized in _SENSITIVE_CONFIG_KEYS or normalized.endswith("_api_key")


def _field(value: Any, name: str, default: Any = None) -> Any:
    return value.get(name, default) if isinstance(value, dict) else getattr(value, name, default)


def _set_field(value: Any, name: str, replacement: Any) -> None:
    if isinstance(value, dict):
        value[name] = replacement
    else:
        setattr(value, name, replacement)


def _canonicalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _canonicalize(nested)
            for key, nested in sorted(value.items(), key=lambda item: str(item[0]))
            if not _is_sensitive_config_key(key)
        }
    if isinstance(value, (list, tuple)):
        return [_canonicalize(item) for item in value]
    if isinstance(value, type):
        schema = getattr(value, "model_json_schema", None)
        return {
            "type": f"{value.__module__}.{value.__qualname__}",
            "schema": _canonicalize(schema()) if callable(schema) else None,
        }
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _canonicalize(model_dump(mode="json"))
    return str(value)


def _zero_argument_tool_names(tools: Any) -> set[str]:
    names: set[str] = set()
    for tool in tools or []:
        function = _field(tool, "function")
        parameters = _field(function, "parameters", {})
        properties = _field(parameters, "properties", {})
        name = _field(function, "name")
        if name and isinstance(properties, dict) and not properties:
            names.add(str(name))
    return names


def _normalize_tool_arguments(response: Any, tools: Any) -> tuple[Any, bool]:
    """Apply one local repair pass only to confirmed zero-argument tools."""
    normalized = copy.deepcopy(response)
    zero_argument_names = _zero_argument_tool_names(tools)
    for choice in _field(normalized, "choices", []) or []:
        message = _field(choice, "message")
        for call in _field(message, "tool_calls", []) or []:
            function = _field(call, "function")
            name = _field(function, "name")
            arguments = _field(function, "arguments")
            if not isinstance(arguments, str):
                return normalized, False
            try:
                parsed = json.loads(arguments)
            except (TypeError, ValueError, json.JSONDecodeError):
                return normalized, False
            if not isinstance(parsed, dict):
                return normalized, False
            if name in zero_argument_names and parsed:
                _set_field(function, "arguments", "{}")
    return normalized, True


def _is_cacheable_response(response: Any, tools: Any) -> bool:
    choices = _field(response, "choices")
    if not isinstance(choices, (list, tuple)) or not choices:
        return False
    tool_calls_seen = 0
    for choice in choices:
        if _field(choice, "finish_reason") == "length":
            return False
        message = _field(choice, "message")
        if message is None:
            return False
        content = _field(message, "content")
        calls = _field(message, "tool_calls", []) or []
        if not content and not calls:
            return False
        for call in calls:
            function = _field(call, "function")
            if not function or not _field(function, "name"):
                return False
            arguments = _field(function, "arguments")
            if not isinstance(arguments, str):
                return False
            try:
                parsed = json.loads(arguments)
            except (TypeError, ValueError, json.JSONDecodeError):
                return False
            if not isinstance(parsed, dict):
                return False
            tool_calls_seen += 1
    return not tools or tool_calls_seen > 0


class CallLimitExceeded(RuntimeError):
    """Raised before a request that would exceed the configured hard cap."""


class RedactedProviderError(RuntimeError):
    """Provider error whose credential-bearing text has been removed."""

    def __init__(
        self,
        message: str,
        status_code: int | None,
        provider_error_code: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.provider_error_code = provider_error_code


def provider_error_code(exc: BaseException) -> str | None:
    """Extract only a provider error code, never failed-generation content."""
    direct = getattr(exc, "provider_error_code", None) or getattr(exc, "code", None)
    if isinstance(direct, str):
        return direct
    body = getattr(exc, "body", None)
    if not isinstance(body, dict):
        return None
    error = body.get("error", body)
    if not isinstance(error, dict):
        return None
    code = error.get("code")
    return code if isinstance(code, str) else None


def is_json_validate_failed(exc: BaseException) -> bool:
    """Match only Groq's retryable strict-schema validation rejection."""
    return (
        LLMRuntime._status_code(exc) == 400
        and provider_error_code(exc) == "json_validate_failed"
    )


@dataclass(slots=True)
class RuntimeStats:
    remote_calls: int = 0
    cache_hits: int = 0
    retry_count: int = 0
    successful_provider_responses: int = 0
    rate_limit_count: int = 0
    empty_response_count: int = 0
    no_tool_call_response_count: int = 0
    tool_call_count: int = 0
    tool_call_counts: dict[str, int] = field(default_factory=dict)
    provider_tool_use_failed_count: int = 0
    unrecovered_error_count: int = 0
    unrecovered_http_statuses: dict[str, int] = field(default_factory=dict)
    call_limit_exceeded_count: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    responses_with_usage: int = 0
    responses_without_usage: int = 0
    first_attempt_count: int = 0
    first_attempt_success_count: int = 0
    json_validate_failed_count: int = 0
    structured_retry_attempt_count: int = 0
    structured_retry_success_count: int = 0
    structured_retry_failure_count: int = 0
    post_retry_completed_decisions: int = 0

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["physical_remote_attempts"] = self.remote_calls
        result["first_attempt_success_rate"] = (
            self.first_attempt_success_count / self.first_attempt_count
            if self.first_attempt_count
            else 0.0
        )
        result["post_retry_success_rate"] = (
            self.post_retry_completed_decisions / self.first_attempt_count
            if self.first_attempt_count
            else 0.0
        )
        result["token_usage_status"] = (
            "available"
            if self.responses_with_usage > 0 and self.responses_without_usage == 0
            else "unavailable"
        )
        return result


class ResponseCache:
    """Small local SQLite response cache; never stores API credentials."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.Lock()
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS llm_response_cache (
                cache_key TEXT PRIMARY KEY,
                payload_kind TEXT NOT NULL,
                payload TEXT NOT NULL,
                created_at REAL NOT NULL
            )
            """
        )
        self._connection.commit()

    def get(self, key: str, *, tools: Any = None) -> Any | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT payload_kind, payload FROM llm_response_cache "
                "WHERE cache_key = ?",
                (key,),
            ).fetchone()
        if row is None:
            return None
        kind, payload = row
        try:
            if kind == "chat_completion":
                try:
                    response = ChatCompletion.model_validate_json(payload)
                except (TypeError, ValueError):
                    # Older Groq responses used ``on_demand``, which newer
                    # OpenAI SDK versions reject. Normalize only at this
                    # persistence boundary and preserve choices/tool calls.
                    decoded = json.loads(payload)
                    if not isinstance(decoded, dict) or "service_tier" not in decoded:
                        raise
                    decoded.pop("service_tier", None)
                    response = ChatCompletion.model_validate(decoded)
            if kind == "json":
                response = json.loads(payload)
            elif kind != "chat_completion":
                self.discard(key)
                return None
        except (TypeError, ValueError, json.JSONDecodeError):
            self.discard(key)
            return None
        response, arguments_valid = _normalize_tool_arguments(response, tools)
        if not arguments_valid or not _is_cacheable_response(response, tools):
            self.discard(key)
            return None
        return response

    def discard(self, key: str) -> None:
        """Remove one unusable entry without affecting the rest of the cache."""
        with self._lock:
            self._connection.execute(
                "DELETE FROM llm_response_cache WHERE cache_key = ?", (key,)
            )
            self._connection.commit()

    def put(self, key: str, response: Any, *, tools: Any = None) -> bool:
        if not _is_cacheable_response(response, tools):
            return False
        if isinstance(response, ChatCompletion):
            kind = "chat_completion"
            payload = response.model_dump_json()
        else:
            try:
                payload = json.dumps(response, ensure_ascii=False, sort_keys=True)
            except (TypeError, ValueError):
                return False
            kind = "json"
        with self._lock:
            self._connection.execute(
                "INSERT OR REPLACE INTO llm_response_cache "
                "(cache_key, payload_kind, payload, created_at) VALUES (?, ?, ?, ?)",
                (key, kind, payload, time.time()),
            )
            self._connection.commit()
        return True


class LLMRuntime:
    """Central hard limits around every remote model invocation."""

    def __init__(
        self,
        settings: LLMSettings,
        *,
        cache: ResponseCache | None = None,
        model_config: dict[str, Any] | None = None,
        sleeper: Callable[[float], None] | None = None,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self.settings = settings
        self.cache = cache
        self.model_config = copy.deepcopy(
            model_config
            if model_config is not None
            else {
                "temperature": settings.temperature,
                "max_tokens": settings.max_tokens,
            }
        )
        self.stats = RuntimeStats()
        self._semaphore = threading.BoundedSemaphore(settings.max_concurrency)
        self._state_lock = threading.Lock()
        self._interval_lock = threading.Lock()
        self._last_request_at: float | None = None
        self._sleep = sleeper or time.sleep
        self._monotonic = monotonic

    def normalize_tool_schemas(self, tools: Any) -> Any:
        """Return provider-compatible copies of OpenAI tool schemas.

        Groq rejects CAMEL's strict zero-argument schemas even when they carry
        an empty ``properties`` object. Preserve strict mode for parameterized
        tools, but disable it for Groq tools that cannot accept arguments.
        """
        if tools is None:
            return None

        normalized = copy.deepcopy(tools)

        def visit(value: Any) -> None:
            if isinstance(value, dict):
                if value.get("type") == "object":
                    value.setdefault("properties", {})
                function = value.get("function")
                if isinstance(function, dict):
                    parameters = function.get("parameters")
                    if (
                        self.settings.provider == "groq"
                        and isinstance(parameters, dict)
                        and not parameters.get("properties")
                    ):
                        function["strict"] = False
                for nested in value.values():
                    visit(nested)
            elif isinstance(value, list):
                for nested in value:
                    visit(nested)

        visit(normalized)
        return normalized

    @staticmethod
    def cache_key(
        *,
        provider: str,
        model: str,
        temperature: float,
        max_tokens: int,
        model_config: dict[str, Any],
        messages: list[dict[str, Any]],
        tools: Any = None,
        tool_choice: Any = None,
        response_format: Any = None,
    ) -> str:
        material = {
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "provider": provider,
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "model_config": _canonicalize(model_config),
            "messages": messages,
            "tools": _canonicalize(tools),
            "tool_choice": _canonicalize(tool_choice),
            "response_format": _canonicalize(response_format),
        }
        encoded = json.dumps(
            material, ensure_ascii=False, sort_keys=True, default=str
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def cache_key_for_request(
        self,
        *,
        messages: list[dict[str, Any]],
        tools: Any = None,
        tool_choice: Any = None,
        response_format: Any = None,
        model_config: dict[str, Any] | None = None,
    ) -> str:
        return self.cache_key(
            provider=self.settings.provider,
            model=self.settings.llm_model,
            temperature=self.settings.temperature,
            max_tokens=self.settings.max_tokens,
            model_config=model_config if model_config is not None else self.model_config,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
        )

    def _reserve_remote_call(self) -> None:
        with self._state_lock:
            if self.stats.remote_calls >= self.settings.max_calls_per_run:
                self.stats.call_limit_exceeded_count += 1
                raise CallLimitExceeded(
                    "LLM_MAX_CALLS_PER_RUN reached; refusing additional remote calls"
                )
            self.stats.remote_calls += 1

    def _respect_interval(self) -> None:
        with self._interval_lock:
            now = self._monotonic()
            if self._last_request_at is not None:
                remaining = (
                    self.settings.request_interval_seconds
                    - (now - self._last_request_at)
                )
                if remaining > 0:
                    self._sleep(remaining)
            self._last_request_at = self._monotonic()

    @staticmethod
    def _status_code(exc: BaseException) -> int | None:
        status = getattr(exc, "status_code", None)
        return status if isinstance(status, int) else None

    @classmethod
    def _is_rate_limit(cls, exc: BaseException) -> bool:
        return cls._status_code(exc) == 429 or "ratelimit" in type(exc).__name__.lower()

    @classmethod
    def _is_transient(cls, exc: BaseException) -> bool:
        status = cls._status_code(exc)
        if status in {408, 429} or (isinstance(status, int) and status >= 500):
            return True
        name = type(exc).__name__.lower()
        return isinstance(exc, TimeoutError) or any(
            marker in name
            for marker in ("timeout", "ratelimit", "connection", "temporar")
        )

    @staticmethod
    def _retry_after_seconds(exc: BaseException) -> float | None:
        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", None) or getattr(exc, "headers", None)
        if headers is None:
            return None
        try:
            raw = headers.get("retry-after") or headers.get("Retry-After")
        except AttributeError:
            return None
        if raw is None:
            return None
        try:
            return max(0.0, float(raw))
        except (TypeError, ValueError):
            try:
                when = parsedate_to_datetime(str(raw))
                if when.tzinfo is None:
                    when = when.replace(tzinfo=timezone.utc)
                return max(0.0, (when - datetime.now(timezone.utc)).total_seconds())
            except (TypeError, ValueError, OverflowError):
                return None

    def _record_response_shape(
        self, response: Any, *, provider: bool, tools_expected: bool
    ) -> None:
        if provider:
            self.stats.successful_provider_responses += 1
        choices = getattr(response, "choices", None)
        if choices is None and isinstance(response, dict):
            choices = response.get("choices")
        if not choices:
            self.stats.empty_response_count += 1
            if tools_expected:
                self.stats.no_tool_call_response_count += 1
            return

        tool_calls = 0
        has_content = False
        for choice in choices:
            message = (
                choice.get("message")
                if isinstance(choice, dict)
                else getattr(choice, "message", None)
            )
            if message is None:
                continue
            content = (
                message.get("content")
                if isinstance(message, dict)
                else getattr(message, "content", None)
            )
            has_content = has_content or bool(content)
            calls = (
                message.get("tool_calls")
                if isinstance(message, dict)
                else getattr(message, "tool_calls", None)
            )
            tool_calls += len(calls or [])
            for call in calls or []:
                function = _field(call, "function")
                name = _field(function, "name")
                if isinstance(name, str):
                    self.stats.tool_call_counts[name] = (
                        self.stats.tool_call_counts.get(name, 0) + 1
                    )
        self.stats.tool_call_count += tool_calls
        if tools_expected and tool_calls == 0:
            self.stats.no_tool_call_response_count += 1
            if not has_content:
                self.stats.empty_response_count += 1

    def _record_unrecovered_error(self, exc: BaseException) -> None:
        self.stats.unrecovered_error_count += 1
        if "tool_use_failed" in str(exc).lower():
            self.stats.provider_tool_use_failed_count += 1
        status = self._status_code(exc)
        if status is not None:
            key = str(status)
            self.stats.unrecovered_http_statuses[key] = (
                self.stats.unrecovered_http_statuses.get(key, 0) + 1
            )

    @property
    def remaining_calls(self) -> int:
        with self._state_lock:
            return max(0, self.settings.max_calls_per_run - self.stats.remote_calls)

    def record_final_error(self, exc: BaseException) -> None:
        """Record one final provider error after bounded correction is exhausted."""
        self._record_unrecovered_error(exc)

    def begin_first_attempt(self) -> None:
        self.stats.first_attempt_count += 1

    def record_json_validate_failed(self) -> None:
        self.stats.json_validate_failed_count += 1

    def begin_structured_retry(self) -> None:
        """Count one correction retry without reserving an extra provider call."""
        if self.remaining_calls <= 0:
            raise CallLimitExceeded(
                "LLM_MAX_CALLS_PER_RUN reached; refusing structured retry"
            )
        self.stats.structured_retry_attempt_count += 1
        self.stats.retry_count += 1

    def record_structured_completion(self, *, retried: bool) -> None:
        if retried:
            self.stats.structured_retry_success_count += 1
        else:
            self.stats.first_attempt_success_count += 1
        self.stats.post_retry_completed_decisions += 1

    def record_structured_retry_failure(self) -> None:
        self.stats.structured_retry_failure_count += 1

    def _redact_provider_exception(self, exc: Exception) -> Exception:
        code = provider_error_code(exc)
        if self._status_code(exc) == 400 and code == "json_validate_failed":
            return RedactedProviderError(
                "provider rejected strict structured output: json_validate_failed",
                400,
                code,
            )
        original = str(exc)
        redacted = original
        for secret in (
            self.settings.groq_api_key,
            self.settings.openai_api_key,
        ):
            if secret:
                redacted = redacted.replace(secret, "[REDACTED]")
        redacted = re.sub(
            r"(?i)\b(?:gsk_|sk-|hf_)[A-Za-z0-9_-]+", "[REDACTED]", redacted
        )
        if redacted == original:
            return exc
        return RedactedProviderError(
            redacted, self._status_code(exc), provider_error_code(exc)
        )

    def _record_usage(self, response: Any) -> None:
        usage = getattr(response, "usage", None)
        if usage is None and isinstance(response, dict):
            usage = response.get("usage")
        if usage is None:
            self.stats.responses_without_usage += 1
            return

        def value(name: str) -> int | None:
            raw = usage.get(name) if isinstance(usage, dict) else getattr(usage, name, None)
            return int(raw) if raw is not None else None

        prompt = value("prompt_tokens")
        completion = value("completion_tokens")
        total = value("total_tokens")
        if prompt is None and completion is None and total is None:
            self.stats.responses_without_usage += 1
            return
        self.stats.responses_with_usage += 1
        self.stats.prompt_tokens += prompt or 0
        self.stats.completion_tokens += completion or 0
        self.stats.total_tokens += total or (prompt or 0) + (completion or 0)

    def call(
        self,
        function: Callable[..., Any],
        *,
        messages: list[dict[str, Any]],
        tools: Any = None,
        tool_choice: Any = None,
        response_format: Any = None,
        model_config: dict[str, Any] | None = None,
        cache_write: bool = True,
        cache_read: bool = True,
        defer_unrecovered_error: bool = False,
    ) -> Any:
        tools = self.normalize_tool_schemas(tools)
        key = self.cache_key_for_request(
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
            model_config=model_config,
        )
        if self.cache is not None and cache_read:
            cached = self.cache.get(key, tools=tools)
            if cached is not None:
                self.stats.cache_hits += 1
                self._record_response_shape(
                    cached, provider=False, tools_expected=bool(tools)
                )
                return cached

        attempt = 0
        while True:
            self._reserve_remote_call()
            self._respect_interval()
            try:
                with self._semaphore:
                    kwargs = {"response_format": response_format, "tools": tools}
                    if tool_choice is not None:
                        kwargs["tool_choice"] = tool_choice
                    response = function(messages, **kwargs)
                response, arguments_valid = _normalize_tool_arguments(response, tools)
                self._record_response_shape(
                    response, provider=True, tools_expected=bool(tools)
                )
                self._record_usage(response)
                if self.cache is not None and arguments_valid and cache_write:
                    self.cache.put(key, response, tools=tools)
                return response
            except Exception as exc:
                if self._is_rate_limit(exc):
                    self.stats.rate_limit_count += 1
                if not self._is_transient(exc) or attempt >= self.settings.max_retries:
                    if not defer_unrecovered_error:
                        self._record_unrecovered_error(exc)
                    safe_exc = self._redact_provider_exception(exc)
                    if safe_exc is exc:
                        raise
                    raise safe_exc from None
                attempt += 1
                self.stats.retry_count += 1
                retry_after = self._retry_after_seconds(exc)
                self._sleep(
                    retry_after
                    if retry_after is not None
                    else float(2 ** (attempt - 1))
                )

    async def acall(
        self,
        function: Callable[..., Any],
        *,
        messages: list[dict[str, Any]],
        tools: Any = None,
        tool_choice: Any = None,
        response_format: Any = None,
        model_config: dict[str, Any] | None = None,
        cache_write: bool = True,
        cache_read: bool = True,
        defer_unrecovered_error: bool = False,
    ) -> Any:
        return await asyncio.to_thread(
            self.call,
            function,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
            model_config=model_config,
            cache_write=cache_write,
            cache_read=cache_read,
            defer_unrecovered_error=defer_unrecovered_error,
        )

    def store_response(
        self,
        response: Any,
        *,
        messages: list[dict[str, Any]],
        response_format: Any,
        model_config: dict[str, Any] | None = None,
    ) -> bool:
        """Persist a response only after local validation and dispatch succeed."""
        if self.cache is None:
            return False
        key = self.cache_key_for_request(
            messages=messages,
            response_format=response_format,
            model_config=model_config,
        )
        return self.cache.put(key, response, tools=None)

    def discard_response(
        self,
        *,
        messages: list[dict[str, Any]],
        response_format: Any,
        model_config: dict[str, Any] | None = None,
    ) -> None:
        """Discard one response that failed validation after cache lookup."""
        if self.cache is None:
            return
        key = self.cache_key_for_request(
            messages=messages,
            response_format=response_format,
            model_config=model_config,
        )
        self.cache.discard(key)


class ManagedModelBackend(BaseModelBackend):
    """CAMEL backend wrapper preserving native tool-call responses."""

    def __init__(self, backend: BaseModelBackend, runtime: LLMRuntime) -> None:
        self.backend = backend
        self.runtime = runtime
        self._backend_config_lock = threading.RLock()
        super().__init__(
            model_type=backend.model_type,
            model_config_dict=backend.model_config_dict,
            token_counter=backend.token_counter,
            timeout=runtime.settings.timeout_seconds,
            max_retries=0,
        )

    @property
    def token_counter(self) -> Any:
        return self.backend.token_counter

    def _run(
        self,
        messages: list[dict[str, Any]],
        response_format: Any = None,
        tools: Any = None,
    ) -> Any:
        tool_choice = (
            "required" if self.runtime.settings.provider == "groq" and tools else None
        )
        return self.runtime.call(
            self._run_backend,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
            model_config=self.backend.model_config_dict,
        )

    def _run_backend(
        self,
        messages: list[dict[str, Any]],
        *,
        response_format: Any = None,
        tools: Any = None,
        tool_choice: Any = None,
    ) -> Any:
        """Inject per-request tool choice without leaking it to later calls."""
        with self._backend_config_lock:
            marker = object()
            previous = self.backend.model_config_dict.get("tool_choice", marker)
            if tool_choice is not None:
                self.backend.model_config_dict["tool_choice"] = tool_choice
            try:
                return self.backend.run(
                    messages, response_format=response_format, tools=tools
                )
            finally:
                if previous is marker:
                    self.backend.model_config_dict.pop("tool_choice", None)
                else:
                    self.backend.model_config_dict["tool_choice"] = previous

    async def _arun(
        self,
        messages: list[dict[str, Any]],
        response_format: Any = None,
        tools: Any = None,
    ) -> Any:
        tool_choice = (
            "required" if self.runtime.settings.provider == "groq" and tools else None
        )
        return await self.runtime.acall(
            self._run_backend,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
            model_config=self.backend.model_config_dict,
        )

    async def structured_arun(
        self,
        messages: list[dict[str, Any]],
        *,
        response_format: dict[str, Any],
        cache_read: bool = True,
        defer_unrecovered_error: bool = False,
    ) -> Any:
        """Send one strict structured request without tools or tool_choice."""
        return await self.runtime.acall(
            self._run_structured_backend,
            messages=messages,
            tools=None,
            tool_choice=None,
            response_format=response_format,
            model_config=self.backend.model_config_dict,
            cache_write=False,
            cache_read=cache_read,
            defer_unrecovered_error=defer_unrecovered_error,
        )

    def _run_structured_backend(
        self,
        messages: list[dict[str, Any]],
        *,
        response_format: dict[str, Any],
        tools: Any = None,
        tool_choice: Any = None,
    ) -> Any:
        if tools is not None or tool_choice is not None:
            raise ValueError("structured action requests cannot include tools")
        custom = getattr(self.backend, "run_strict_structured", None)
        if callable(custom):
            return custom(messages, response_format=response_format)

        prepare = getattr(self.backend, "_prepare_request_config", None)
        call_client = getattr(self.backend, "_call_client", None)
        client = getattr(self.backend, "_client", None)
        if not callable(prepare) or not callable(call_client) or client is None:
            raise TypeError(
                "backend cannot issue exact strict structured-output requests"
            )
        request_config = prepare(None)
        for key in ("tools", "tool_choice", "response_format", "stream"):
            request_config.pop(key, None)
        request_config["response_format"] = response_format
        return call_client(
            client.chat.completions.create,
            messages=messages,
            model=self.backend.model_type,
            **request_config,
        )

    def cache_structured_response(
        self,
        response: Any,
        *,
        messages: list[dict[str, Any]],
        response_format: dict[str, Any],
    ) -> bool:
        return self.runtime.store_response(
            response,
            messages=messages,
            response_format=response_format,
            model_config=self.backend.model_config_dict,
        )

    def discard_structured_response(
        self,
        *,
        messages: list[dict[str, Any]],
        response_format: dict[str, Any],
    ) -> None:
        self.runtime.discard_response(
            messages=messages,
            response_format=response_format,
            model_config=self.backend.model_config_dict,
        )
