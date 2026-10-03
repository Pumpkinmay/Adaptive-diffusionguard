"""Minimal Transformers/PEFT LoRA and QLoRA training entry point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import tomllib

from training.schemas import BehaviorExample


def load_examples(path: Path) -> list[BehaviorExample]:
    examples: list[BehaviorExample] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                examples.append(BehaviorExample.from_dict(json.loads(line)))
            except Exception as exc:
                raise ValueError(f"invalid sample on line {line_number}: {exc}") from exc
    if not examples:
        raise ValueError("dataset is empty")
    return examples


def load_config(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        config = tomllib.load(handle)
    if not config.get("model", {}).get("name"):
        raise ValueError("config requires [model].name")
    return config


def build_masked_features(
    tokenizer: Any,
    prompt: str,
    target: str,
    max_length: int,
) -> dict[str, list[int]]:
    """Tokenize prompt/target and mask prompt tokens from training loss."""
    prompt_ids = list(
        tokenizer(prompt, add_special_tokens=True, truncation=False)["input_ids"]
    )
    target_ids = list(
        tokenizer(target, add_special_tokens=False, truncation=False)["input_ids"]
    )
    eos = getattr(tokenizer, "eos_token_id", None)
    if eos is not None:
        target_ids.append(int(eos))
    if not target_ids:
        raise ValueError("target tokenization is empty")
    target_ids = target_ids[:max_length]
    prompt_budget = max(0, max_length - len(target_ids))
    prompt_ids = prompt_ids[-prompt_budget:] if prompt_budget else []
    input_ids = prompt_ids + target_ids
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": [-100] * len(prompt_ids) + target_ids,
    }


class _DryRunTokenizer:
    eos_token_id = 0

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        truncation: bool,
    ) -> dict[str, list[int]]:
        del truncation
        tokens = [index + 2 for index, _ in enumerate(text.split())]
        if add_special_tokens:
            tokens.insert(0, 1)
        return {"input_ids": tokens}


def dry_run(examples: list[BehaviorExample], config: dict[str, Any], output: Path) -> None:
    max_length = int(config.get("training", {}).get("max_length", 512))
    tokenizer = _DryRunTokenizer()
    features = [
        build_masked_features(
            tokenizer, example.prompt(), example.label.to_json(), max_length
        )
        for example in examples
    ]
    if any(-100 not in item["labels"] for item in features):
        raise ValueError("prompt label masking validation failed")
    if any(not any(value != -100 for value in item["labels"]) for item in features):
        raise ValueError("target label validation failed")
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "mode": "dry-run",
        "validated_samples": len(examples),
        "tokenized_samples": len(features),
        "prompt_tokens_masked": True,
        "output_schema_validated": True,
        "model_name": config["model"]["name"],
        "label_sources": sorted({example.label_source for example in examples}),
        "adapter_saved": False,
        "note": "No model was downloaded and no optimization was performed.",
    }
    (output / "dry_run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False))


def train(examples: list[BehaviorExample], config: dict[str, Any], output: Path) -> None:
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            DataCollatorForSeq2Seq,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise SystemExit(
            "training extras are missing; install with `uv sync --extra training`"
        ) from exc

    model_cfg = config["model"]
    train_cfg = config.get("training", {})
    lora_cfg = config.get("lora", {})
    use_qlora = bool(model_cfg.get("qlora", False))
    quantization = None
    if use_qlora:
        if not torch.cuda.is_available():
            raise RuntimeError("QLoRA requires a supported CUDA environment")
        quantization = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["name"], quantization_config=quantization
    )
    if use_qlora:
        model = prepare_model_for_kbit_training(model)
    model = get_peft_model(
        model,
        LoraConfig(
            r=int(lora_cfg.get("r", 8)),
            lora_alpha=int(lora_cfg.get("alpha", 16)),
            lora_dropout=float(lora_cfg.get("dropout", 0.05)),
            target_modules=lora_cfg.get("target_modules"),
            task_type="CAUSAL_LM",
        ),
    )
    max_length = int(train_cfg.get("max_length", 512))
    dataset = Dataset.from_list(
        [
            {"prompt": example.prompt(), "target": example.label.to_json()}
            for example in examples
        ]
    )

    def tokenize(row: dict[str, str]) -> dict[str, list[int]]:
        return build_masked_features(
            tokenizer, row["prompt"], row["target"], max_length
        )

    tokenized = dataset.map(tokenize, remove_columns=["prompt", "target"])
    args = TrainingArguments(
        output_dir=str(output / "checkpoints"),
        num_train_epochs=float(train_cfg.get("epochs", 1)),
        per_device_train_batch_size=int(train_cfg.get("batch_size", 1)),
        gradient_accumulation_steps=int(train_cfg.get("gradient_accumulation", 1)),
        learning_rate=float(train_cfg.get("learning_rate", 2e-4)),
        logging_steps=1,
        save_strategy="no",
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, padding=True, label_pad_token_id=-100
        ),
    )
    trainer.train()
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output / "adapter")
    tokenizer.save_pretrained(output / "adapter")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    examples = load_examples(args.dataset)
    if args.dry_run:
        dry_run(examples, config, args.output)
    else:
        train(examples, config, args.output)


if __name__ == "__main__":
    main()
