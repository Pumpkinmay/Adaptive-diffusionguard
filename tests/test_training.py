import json

import pytest

from training.build_dataset import demo_examples, write_jsonl
from training.evaluate_actions import evaluate
from training.schemas import ActionOutput
from training.train_lora import build_masked_features, load_examples


def test_action_output_is_strict_json() -> None:
    parsed = ActionOutput.parse_json(
        '{"action":"report","confidence":0.8,"reason":"warning"}'
    )
    assert parsed.action == "report"
    with pytest.raises(ValueError):
        ActionOutput.parse_json(
            '{"action":"like","confidence":0.8,"reason":"not allowed"}'
        )
    with pytest.raises(ValueError):
        ActionOutput.parse_json(
            '{"action":"ignore","confidence":0.8,"reason":"ok","extra":1}'
        )


def test_demo_dataset_and_metrics(tmp_path) -> None:
    dataset = tmp_path / "actions.jsonl"
    assert write_jsonl(demo_examples("teacher_synthetic"), dataset) == 4
    examples = load_examples(dataset)
    assert len(examples) == 4
    assert {item.label_source for item in examples} == {"teacher_synthetic"}

    rows = [
        {
            "label": "report",
            "output": json.dumps(
                {"action": "report", "confidence": 0.8, "reason": "x"}
            ),
            "latency_seconds": 0.2,
        },
        {"label": "ignore", "output": "not json", "latency_seconds": 0.4},
    ]
    metrics = evaluate(rows)
    assert metrics["json_valid_rate"] == 0.5
    assert metrics["mean_inference_seconds"] == pytest.approx(0.3)
    assert set(metrics) == {
        "action_macro_f1",
        "json_valid_rate",
        "brier_score",
        "mean_inference_seconds",
    }


def test_prompt_tokens_are_masked_from_training_loss() -> None:
    class Tokenizer:
        eos_token_id = 9

        def __call__(self, text, *, add_special_tokens, truncation):
            del text, truncation
            return {"input_ids": [1, 2] if add_special_tokens else [3, 4]}

    features = build_masked_features(Tokenizer(), "prompt", "target", 10)
    assert features["input_ids"] == [1, 2, 3, 4, 9]
    assert features["labels"] == [-100, -100, 3, 4, 9]
