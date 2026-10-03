from adaptive_diffusionguard.experiments.real_llm_validation import (
    EXIT_CODES,
    _determine_status,
)


def runtime_stats(**overrides):
    values = {
        "successful_provider_responses": 15,
        "physical_remote_attempts": 15,
        "max_calls_per_run": 30,
        "unrecovered_http_statuses": {},
        "call_limit_exceeded_count": 0,
        "empty_response_count": 0,
        "no_tool_call_response_count": 0,
        "unrecovered_error_count": 0,
    }
    values.update(overrides)
    return values


def test_complete_validation_is_success() -> None:
    status, reasons = _determine_status(
        logical_decisions=15,
        completed_decisions=15,
        impression_count=30,
        governance_executed=True,
        secret_scan_passed=True,
        runtime=runtime_stats(),
    )
    assert status == "success"
    assert EXIT_CODES[status] == 0
    assert reasons == []


def test_partial_rate_limited_validation_is_degraded() -> None:
    status, reasons = _determine_status(
        logical_decisions=15,
        completed_decisions=4,
        impression_count=30,
        governance_executed=True,
        secret_scan_passed=True,
        runtime=runtime_stats(
            successful_provider_responses=9,
            physical_remote_attempts=30,
            unrecovered_http_statuses={"429": 3},
            call_limit_exceeded_count=3,
            empty_response_count=2,
            no_tool_call_response_count=5,
            unrecovered_error_count=3,
        ),
    )
    assert status == "degraded"
    assert EXIT_CODES[status] != 0
    assert "incomplete_agent_decisions" in reasons
    assert "unrecovered_http_429" in reasons
    assert "physical_call_limit_reached_with_incomplete_decisions" in reasons


def test_no_provider_response_or_impressions_is_failed() -> None:
    status, reasons = _determine_status(
        logical_decisions=15,
        completed_decisions=0,
        impression_count=0,
        governance_executed=False,
        secret_scan_passed=True,
        runtime=runtime_stats(successful_provider_responses=0),
    )
    assert status == "failed"
    assert EXIT_CODES[status] != 0
    assert "no_successful_provider_responses" in reasons
    assert "no_impression_records" in reasons
