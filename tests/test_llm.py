import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any

import pytest
from camel.models import BaseModelBackend
from camel.types import ModelPlatformType

from adaptive_diffusionguard.llm.actions import normalize_action
from adaptive_diffusionguard.llm.model_factory import (
    LLMConfigurationError,
    LLMSettings,
    create_llm_model,
)
from adaptive_diffusionguard.llm.runtime import (
    CallLimitExceeded,
    LLMRuntime,
    ManagedModelBackend,
    ResponseCache,
)
from adaptive_diffusionguard.llm.structured_actions import (
    ActionMask,
    build_response_format,
    parse_structured_response,
)


class FakeBackend(BaseModelBackend):
    def __init__(self, response: Any | None = None) -> None:
        self.response = response or {"choices": []}
        self._counter = object()
        super().__init__(model_type="fake-model", model_config_dict={})

    @property
    def token_counter(self) -> Any:
        return self._counter

    def _run(self, messages, response_format=None, tools=None):
        return self.response

    async def _arun(self, messages, response_format=None, tools=None):
        return self.response


def settings(**overrides: Any) -> LLMSettings:
    values = {
        "enabled": False,
        "request_interval_seconds": 0.0,
        "cache_enabled": False,
    }
    values.update(overrides)
    return LLMSettings(**values)


def content_response(*, finish_reason: str = "stop") -> dict[str, Any]:
    return {
        "choices": [
            {
                "finish_reason": finish_reason,
                "message": {"content": "ok", "tool_calls": None},
            }
        ]
    }


def tool_response(
    name: str = "do_nothing",
    arguments: str = "{}",
    *,
    finish_reason: str = "tool_calls",
) -> dict[str, Any]:
    return {
        "choices": [
            {
                "finish_reason": finish_reason,
                "message": {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": name, "arguments": arguments},
                        }
                    ],
                },
            }
        ]
    }


ZERO_ARGUMENT_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "do_nothing",
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    }
]
PARAMETERIZED_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "repost",
            "parameters": {
                "type": "object",
                "properties": {"post_id": {"type": "integer"}},
                "required": ["post_id"],
            },
        },
    }
]


def test_disabled_llm_does_not_create_model() -> None:
    called = False

    def factory(**kwargs):
        nonlocal called
        called = True
        return FakeBackend()

    assert create_llm_model(settings(), factory=factory) is None
    assert called is False


def test_groq_requires_groq_key() -> None:
    with pytest.raises(LLMConfigurationError, match="GROQ_API_KEY"):
        create_llm_model(settings(enabled=True, provider="groq"))


def test_openai_requires_openai_key() -> None:
    with pytest.raises(LLMConfigurationError, match="OPENAI_API_KEY"):
        create_llm_model(settings(enabled=True, provider="openai"))


def test_unsupported_provider_is_rejected() -> None:
    with pytest.raises(LLMConfigurationError, match="unsupported LLM_PROVIDER"):
        create_llm_model(settings(provider="other"))


def test_groq_passes_string_model_and_config() -> None:
    captured = {}

    def factory(**kwargs):
        captured.update(kwargs)
        return FakeBackend()

    model = create_llm_model(
        settings(
            enabled=True,
            provider="groq",
            groq_api_key="test-secret",
            llm_model="openai/gpt-oss-20b",
            temperature=0.2,
            max_tokens=123,
            timeout_seconds=7,
            max_retries=4,
        ),
        factory=factory,
    )
    assert model is not None
    assert captured["model_platform"] is ModelPlatformType.GROQ
    assert captured["model_type"] == "openai/gpt-oss-20b"
    assert captured["api_key"] == "test-secret"
    assert captured["model_config_dict"] == {"temperature": 0.2, "max_tokens": 123}
    assert captured["timeout"] == 7
    assert captured["max_retries"] == 0


def test_429_retries_are_limited() -> None:
    class RateLimited(Exception):
        status_code = 429

    attempts = 0

    def fail(messages, response_format=None, tools=None):
        nonlocal attempts
        attempts += 1
        raise RateLimited("temporary")

    runtime = LLMRuntime(settings(max_retries=2, max_calls_per_run=10), sleeper=lambda _: None)
    with pytest.raises(RateLimited):
        runtime.call(fail, messages=[])
    assert attempts == 3
    assert runtime.stats.retry_count == 2
    assert runtime.stats.rate_limit_count == 3
    assert runtime.stats.remote_calls == 3


def test_retry_after_header_takes_precedence() -> None:
    class Response:
        def __init__(self) -> None:
            self.headers = {"Retry-After": "7"}

    class RateLimited(Exception):
        status_code = 429

        def __init__(self, message: str) -> None:
            super().__init__(message)
            self.response = Response()

    attempts = 0
    sleeps: list[float] = []

    def eventually_succeed(messages, response_format=None, tools=None):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RateLimited("temporary")
        return {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [{"id": "one"}],
                    }
                }
            ]
        }

    runtime = LLMRuntime(
        settings(max_retries=1, max_calls_per_run=2), sleeper=sleeps.append
    )
    runtime.call(eventually_succeed, messages=[])

    assert sleeps == [7.0]
    assert runtime.stats.remote_calls == 2
    assert runtime.stats.retry_count == 1
    assert runtime.stats.rate_limit_count == 1
    assert runtime.stats.successful_provider_responses == 1
    assert runtime.stats.tool_call_count == 1


def test_conservative_environment_defaults() -> None:
    value = LLMSettings.from_env({})
    assert value.max_retries == 1
    assert value.max_concurrency == 1
    assert value.max_calls_per_run == 30
    assert value.request_interval_seconds == 8.0


def test_groq_zero_argument_tool_schema_gets_properties() -> None:
    captured = {}

    def succeed(messages, response_format=None, tools=None):
        captured["tools"] = tools
        return {"ok": True}

    runtime = LLMRuntime(settings(), sleeper=lambda _: None)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "refresh",
                "strict": True,
                "parameters": {"type": "object", "required": []},
            },
        }
    ]
    runtime.call(succeed, messages=[], tools=tools)

    parameters = captured["tools"][0]["function"]["parameters"]
    assert parameters == {"type": "object", "required": [], "properties": {}}
    assert captured["tools"][0]["function"]["strict"] is False
    assert "properties" not in tools[0]["function"]["parameters"]
    assert tools[0]["function"]["strict"] is True


def test_cache_hit_avoids_second_remote_call(tmp_path: Path) -> None:
    calls = 0

    def succeed(messages, response_format=None, tools=None):
        nonlocal calls
        calls += 1
        return content_response()

    runtime = LLMRuntime(
        settings(cache_enabled=True),
        cache=ResponseCache(tmp_path / "cache.db"),
        sleeper=lambda _: None,
    )
    first = runtime.call(succeed, messages=[{"role": "system", "content": "s"}])
    second = runtime.call(succeed, messages=[{"role": "system", "content": "s"}])
    assert first == second == content_response()
    assert calls == 1
    assert runtime.stats.cache_hits == 1


def test_max_tokens_change_cannot_hit_previous_cache(tmp_path: Path) -> None:
    cache = ResponseCache(tmp_path / "cache.db")
    calls = 0

    def succeed(messages, response_format=None, tools=None):
        nonlocal calls
        calls += 1
        return content_response()

    messages = [{"role": "user", "content": "same input"}]
    first = LLMRuntime(
        settings(cache_enabled=True, max_tokens=256), cache=cache, sleeper=lambda _: None
    )
    second = LLMRuntime(
        settings(cache_enabled=True, max_tokens=512), cache=cache, sleeper=lambda _: None
    )
    first.call(succeed, messages=messages)
    second.call(succeed, messages=messages)
    assert calls == 2
    assert second.stats.cache_hits == 0


def test_cache_key_covers_provider_model_config_tools_and_formats() -> None:
    runtime = LLMRuntime(settings(), model_config={"top_p": 0.9})
    base = runtime.cache_key_for_request(
        messages=[],
        tools=ZERO_ARGUMENT_TOOL,
        tool_choice="required",
        response_format={"type": "json_object"},
    )
    assert base != LLMRuntime(
        settings(provider="openai"), model_config={"top_p": 0.9}
    ).cache_key_for_request(
        messages=[],
        tools=ZERO_ARGUMENT_TOOL,
        tool_choice="required",
        response_format={"type": "json_object"},
    )
    assert base != LLMRuntime(
        settings(llm_model="different-model"), model_config={"top_p": 0.9}
    ).cache_key_for_request(
        messages=[],
        tools=ZERO_ARGUMENT_TOOL,
        tool_choice="required",
        response_format={"type": "json_object"},
    )
    assert base != LLMRuntime(
        settings(temperature=0.7), model_config={"top_p": 0.9}
    ).cache_key_for_request(
        messages=[],
        tools=ZERO_ARGUMENT_TOOL,
        tool_choice="required",
        response_format={"type": "json_object"},
    )
    assert base != runtime.cache_key_for_request(
        messages=[],
        tools=ZERO_ARGUMENT_TOOL,
        tool_choice="auto",
        response_format={"type": "json_object"},
    )
    assert base != runtime.cache_key_for_request(
        messages=[],
        tools=PARAMETERIZED_TOOL,
        tool_choice="required",
        response_format={"type": "json_object"},
    )
    assert base != runtime.cache_key_for_request(
        messages=[],
        tools=ZERO_ARGUMENT_TOOL,
        tool_choice="required",
        response_format={"type": "text"},
    )
    assert base != runtime.cache_key_for_request(
        messages=[],
        tools=ZERO_ARGUMENT_TOOL,
        tool_choice="required",
        response_format={"type": "json_object"},
        model_config={"top_p": 0.8},
    )


def test_length_response_is_not_cached(tmp_path: Path) -> None:
    calls = 0

    def truncated(messages, response_format=None, tools=None):
        nonlocal calls
        calls += 1
        return content_response(finish_reason="length")

    runtime = LLMRuntime(
        settings(cache_enabled=True),
        cache=ResponseCache(tmp_path / "cache.db"),
        sleeper=lambda _: None,
    )
    runtime.call(truncated, messages=[])
    runtime.call(truncated, messages=[])
    assert calls == 2
    assert runtime.stats.cache_hits == 0


def test_tools_without_tool_calls_are_not_cached(tmp_path: Path) -> None:
    calls = 0

    def text_only(messages, response_format=None, tools=None):
        nonlocal calls
        calls += 1
        return content_response()

    runtime = LLMRuntime(
        settings(cache_enabled=True),
        cache=ResponseCache(tmp_path / "cache.db"),
        sleeper=lambda _: None,
    )
    runtime.call(text_only, messages=[], tools=ZERO_ARGUMENT_TOOL)
    runtime.call(text_only, messages=[], tools=ZERO_ARGUMENT_TOOL)
    assert calls == 2
    assert runtime.stats.cache_hits == 0


@pytest.mark.parametrize(
    "response",
    [
        {},
        {"choices": []},
        {"choices": [{"finish_reason": "stop", "message": None}]},
        tool_response("repost", "not-json"),
    ],
)
def test_empty_or_invalid_argument_response_is_not_cached(
    tmp_path: Path, response: dict[str, Any]
) -> None:
    calls = 0

    def invalid(messages, response_format=None, tools=None):
        nonlocal calls
        calls += 1
        return response

    runtime = LLMRuntime(
        settings(cache_enabled=True),
        cache=ResponseCache(tmp_path / "cache.db"),
        sleeper=lambda _: None,
    )
    runtime.call(invalid, messages=[])
    runtime.call(invalid, messages=[])
    assert calls == 2
    assert runtime.stats.cache_hits == 0


def test_invalid_legacy_cache_is_deleted_and_request_runs(tmp_path: Path) -> None:
    cache = ResponseCache(tmp_path / "cache.db")
    runtime = LLMRuntime(
        settings(cache_enabled=True), cache=cache, sleeper=lambda _: None
    )
    normalized_tools = runtime.normalize_tool_schemas(ZERO_ARGUMENT_TOOL)
    key = runtime.cache_key_for_request(
        messages=[], tools=normalized_tools, tool_choice="required"
    )
    with sqlite3.connect(cache.path) as connection:
        connection.execute(
            "INSERT INTO llm_response_cache "
            "(cache_key, payload_kind, payload, created_at) VALUES (?, ?, ?, ?)",
            (key, "json", json.dumps(content_response(finish_reason="length")), time.time()),
        )
    calls = 0

    def valid(messages, response_format=None, tools=None, tool_choice=None):
        nonlocal calls
        calls += 1
        return tool_response()

    result = runtime.call(
        valid, messages=[], tools=ZERO_ARGUMENT_TOOL, tool_choice="required"
    )
    assert calls == 1
    assert result == tool_response()
    assert runtime.stats.cache_hits == 0


def test_valid_tool_call_response_is_cached(tmp_path: Path) -> None:
    calls = 0

    def valid(messages, response_format=None, tools=None, tool_choice=None):
        nonlocal calls
        calls += 1
        return tool_response()

    runtime = LLMRuntime(
        settings(cache_enabled=True),
        cache=ResponseCache(tmp_path / "cache.db"),
        sleeper=lambda _: None,
    )
    first = runtime.call(
        valid, messages=[], tools=ZERO_ARGUMENT_TOOL, tool_choice="required"
    )
    second = runtime.call(
        valid, messages=[], tools=ZERO_ARGUMENT_TOOL, tool_choice="required"
    )
    assert first == second == tool_response()
    assert calls == 1
    assert runtime.stats.cache_hits == 1


def test_zero_argument_tool_extras_are_normalized_once(tmp_path: Path) -> None:
    runtime = LLMRuntime(settings(), sleeper=lambda _: None)
    result = runtime.call(
        lambda *args, **kwargs: tool_response(
            "do_nothing", '{"action":"do_nothing"}'
        ),
        messages=[],
        tools=ZERO_ARGUMENT_TOOL,
        tool_choice="required",
    )
    arguments = result["choices"][0]["message"]["tool_calls"][0]["function"][
        "arguments"
    ]
    assert arguments == "{}"


def test_parameterized_tool_arguments_are_preserved() -> None:
    runtime = LLMRuntime(settings(), sleeper=lambda _: None)
    result = runtime.call(
        lambda *args, **kwargs: tool_response("repost", '{"post_id":7}'),
        messages=[],
        tools=PARAMETERIZED_TOOL,
        tool_choice="required",
    )
    arguments = result["choices"][0]["message"]["tool_calls"][0]["function"][
        "arguments"
    ]
    assert arguments == '{"post_id":7}'


def test_managed_groq_backend_requires_tool_choice_only_for_tool_calls() -> None:
    backend = FakeBackend(tool_response())
    seen: list[Any] = []
    original_run = backend.run

    def capture(messages, response_format=None, tools=None):
        seen.append(backend.model_config_dict.get("tool_choice"))
        return original_run(messages, response_format=response_format, tools=tools)

    backend.run = capture
    runtime = LLMRuntime(settings(provider="groq"), sleeper=lambda _: None)
    managed = ManagedModelBackend(backend, runtime)
    managed.run([], tools=ZERO_ARGUMENT_TOOL)
    assert seen == ["required"]
    assert "tool_choice" not in backend.model_config_dict


def test_legacy_cache_normalizes_unknown_service_tier_and_preserves_tools(
    tmp_path: Path,
) -> None:
    cache = ResponseCache(tmp_path / "cache.db")
    payload = {
        "id": "chatcmpl-legacy",
        "choices": [
            {
                "finish_reason": "tool_calls",
                "index": 0,
                "logprobs": None,
                "message": {
                    "content": None,
                    "refusal": None,
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "function": {
                                "arguments": '{"post_id":1}',
                                "name": "repost",
                            },
                            "type": "function",
                        }
                    ],
                },
            }
        ],
        "created": 1,
        "model": "openai/gpt-oss-20b",
        "object": "chat.completion",
        "service_tier": "on_demand",
    }
    with sqlite3.connect(cache.path) as connection:
        connection.execute(
            "INSERT INTO llm_response_cache "
            "(cache_key, payload_kind, payload, created_at) VALUES (?, ?, ?, ?)",
            ("legacy", "chat_completion", json.dumps(payload), time.time()),
        )

    response = cache.get("legacy")

    assert response is not None
    assert response.service_tier is None
    tool_call = response.choices[0].message.tool_calls[0]
    assert tool_call.function.name == "repost"
    assert tool_call.function.arguments == '{"post_id":1}'


def test_incompatible_cache_entry_is_a_miss(tmp_path: Path) -> None:
    cache = ResponseCache(tmp_path / "cache.db")
    with sqlite3.connect(cache.path) as connection:
        connection.execute(
            "INSERT INTO llm_response_cache "
            "(cache_key, payload_kind, payload, created_at) VALUES (?, ?, ?, ?)",
            ("broken", "chat_completion", "not-json", time.time()),
        )

    assert cache.get("broken") is None


def test_maximum_call_count_is_hard_limit() -> None:
    runtime = LLMRuntime(settings(max_calls_per_run=1), sleeper=lambda _: None)
    runtime.call(lambda *args, **kwargs: {"ok": True}, messages=[])
    with pytest.raises(CallLimitExceeded, match="LLM_MAX_CALLS_PER_RUN"):
        runtime.call(lambda *args, **kwargs: {"ok": True}, messages=[])


def test_invalid_json_and_action_fall_back_safely() -> None:
    invalid_json = normalize_action("not json")
    invalid_action = normalize_action(
        '{"action":"like","confidence":0.9,"reason":"bad"}',
        repair=lambda _: "still not json",
    )
    assert invalid_json.output.action == "ignore"
    assert invalid_json.action_valid is False
    assert invalid_action.output.action == "ignore"
    assert invalid_action.repair_attempted is True
    assert invalid_action.action_valid is False


def test_api_key_is_not_in_repr_or_logs(caplog) -> None:
    secret = "test-secret-never-log"
    value = settings(enabled=True, provider="groq", groq_api_key=secret)
    with caplog.at_level(logging.DEBUG):
        logging.getLogger("test").debug("settings=%r", value)
    assert secret not in repr(value)
    assert secret not in caplog.text


def test_api_key_is_redacted_from_provider_exception() -> None:
    secret = "test-secret-never-emit"
    runtime = LLMRuntime(
        settings(groq_api_key=secret, max_retries=0), sleeper=lambda _: None
    )

    def fail(messages, response_format=None, tools=None):
        raise RuntimeError(f"provider rejected credential {secret}")

    with pytest.raises(RuntimeError) as caught:
        runtime.call(fail, messages=[])

    assert secret not in str(caught.value)
    assert "[REDACTED]" in str(caught.value)


def test_runtime_counts_calls_by_tool_name() -> None:
    runtime = LLMRuntime(settings(), sleeper=lambda _: None)

    runtime.call(
        lambda *args, **kwargs: tool_response(
            "quote", "{\"post_id\":7,\"text\":\"x\"}"
        ),
        messages=[],
        tools=PARAMETERIZED_TOOL,
        tool_choice="required",
    )

    assert runtime.stats.tool_call_count == 1
    assert runtime.stats.tool_call_counts == {"quote": 1}


def test_unrecovered_tool_use_failed_is_counted() -> None:
    class ToolUseFailed(Exception):
        status_code = 400

    runtime = LLMRuntime(settings(max_retries=0), sleeper=lambda _: None)

    def fail(messages, response_format=None, tools=None):
        raise ToolUseFailed("provider code tool_use_failed")

    with pytest.raises(ToolUseFailed):
        runtime.call(fail, messages=[], tools=PARAMETERIZED_TOOL)

    assert runtime.stats.provider_tool_use_failed_count == 1
    assert runtime.stats.unrecovered_http_statuses == {"400": 1}


def structured_response(
    choice_id: str,
    rationale: str = "reason",
    *,
    finish_reason: str = "stop",
) -> dict[str, Any]:
    return {
        "choices": [
            {
                "finish_reason": finish_reason,
                "message": {
                    "content": json.dumps(
                        {"choice_id": choice_id, "rationale": rationale}
                    ),
                    "tool_calls": None,
                },
            }
        ]
    }


def test_dynamic_response_format_enum_changes_cache_key(tmp_path: Path) -> None:
    calls = 0

    def respond(messages, response_format=None, tools=None):
        nonlocal calls
        calls += 1
        return structured_response("ignore")

    runtime = LLMRuntime(
        settings(cache_enabled=True),
        cache=ResponseCache(tmp_path / "cache.db"),
        sleeper=lambda _: None,
    )
    messages = [{"role": "user", "content": "decide"}]
    first_format = build_response_format(ActionMask(("ignore", "quote:1")))
    changed_format = build_response_format(ActionMask(("ignore", "quote:2")))

    first = runtime.call(
        respond,
        messages=messages,
        response_format=first_format,
        cache_write=False,
    )
    assert runtime.store_response(
        first, messages=messages, response_format=first_format
    )
    runtime.call(
        respond,
        messages=messages,
        response_format=first_format,
        cache_write=False,
    )
    runtime.call(
        respond,
        messages=messages,
        response_format=changed_format,
        cache_write=False,
    )

    assert calls == 2
    assert runtime.stats.cache_hits == 1


def test_invalid_structured_response_is_not_cached(tmp_path: Path) -> None:
    calls = 0

    def invalid(messages, response_format=None, tools=None):
        nonlocal calls
        calls += 1
        return structured_response("not-in-mask", rationale="")

    runtime = LLMRuntime(
        settings(cache_enabled=True),
        cache=ResponseCache(tmp_path / "cache.db"),
        sleeper=lambda _: None,
    )
    messages = [{"role": "user", "content": "decide"}]
    response_format = build_response_format(ActionMask(("ignore",)))
    for _ in range(2):
        response = runtime.call(
            invalid,
            messages=messages,
            response_format=response_format,
            cache_write=False,
        )
        with pytest.raises(ValueError):
            parse_structured_response(response)

    assert calls == 2
    assert runtime.stats.cache_hits == 0


def test_truncated_structured_response_is_rejected_and_not_cached(
    tmp_path: Path,
) -> None:
    calls = 0

    def truncated(messages, response_format=None, tools=None):
        nonlocal calls
        calls += 1
        return structured_response("ignore", finish_reason="length")

    runtime = LLMRuntime(
        settings(cache_enabled=True),
        cache=ResponseCache(tmp_path / "cache.db"),
        sleeper=lambda _: None,
    )
    response_format = build_response_format(ActionMask(("ignore",)))
    for _ in range(2):
        response = runtime.call(
            truncated,
            messages=[],
            response_format=response_format,
            cache_write=False,
        )
        with pytest.raises(ValueError, match="truncated"):
            parse_structured_response(response)

    assert calls == 2
    assert runtime.stats.cache_hits == 0


def test_invalid_structured_cache_entry_is_discarded(tmp_path: Path) -> None:
    calls = 0

    def valid(messages, response_format=None, tools=None):
        nonlocal calls
        calls += 1
        return structured_response("ignore")

    runtime = LLMRuntime(
        settings(cache_enabled=True),
        cache=ResponseCache(tmp_path / "cache.db"),
        sleeper=lambda _: None,
    )
    messages = [{"role": "user", "content": "decide"}]
    response_format = build_response_format(ActionMask(("ignore",)))
    invalid = structured_response("not-in-mask", rationale="")
    assert runtime.store_response(
        invalid, messages=messages, response_format=response_format
    )

    cached = runtime.call(
        valid,
        messages=messages,
        response_format=response_format,
        cache_write=False,
    )
    with pytest.raises(ValueError):
        parse_structured_response(cached)
    runtime.discard_response(
        messages=messages, response_format=response_format
    )
    response = runtime.call(
        valid,
        messages=messages,
        response_format=response_format,
        cache_write=False,
    )

    assert parse_structured_response(response).choice_id == "ignore"
    assert calls == 1




def test_structured_cache_key_never_contains_api_key() -> None:
    secret = "credential-must-not-be-key-material"
    runtime = LLMRuntime(
        settings(groq_api_key=secret),
        model_config={"temperature": 0.0, "api_key": secret},
    )
    key = runtime.cache_key_for_request(
        messages=[{"role": "user", "content": "safe"}],
        response_format=build_response_format(ActionMask(("ignore",))),
    )
    assert secret not in key
    assert len(key) == 64
