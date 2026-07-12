#!/usr/bin/env python3
"""Run a small local GPT-2 persona fine-tune under model-lane ownership."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_text(example: dict[str, Any]) -> str:
    user = example.get("user") or example.get("prompt") or ""
    assistant = (
        example.get("assistant")
        or example.get("generated")
        or example.get("response")
        or ""
    )
    if isinstance(assistant, str) and assistant.startswith("<coroutine"):
        assistant = ""
    return f"{user}\n{assistant}".strip()


def _run_training(args: argparse.Namespace, texts: list[str]) -> None:
    from datasets import Dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    dataset = Dataset.from_dict({"text": texts})
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_fn(example: dict[str, Any]) -> Any:
        return tokenizer(example["text"], truncation=True, max_length=512)

    tokenized = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    model = AutoModelForCausalLM.from_pretrained(args.model)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        save_total_limit=2,
        logging_steps=10,
        fp16=False,
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized,
    )

    print(f"Starting training: {len(texts)} examples, epochs={args.epochs}")
    trainer.train()
    trainer.save_model(args.output_dir)
    print(f"Saved fine-tuned model to {args.output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-file",
        default="data/personality_training/starter_aura.jsonl",
    )
    parser.add_argument("--model", default="distilgpt2")
    parser.add_argument("--output-dir", default="outputs/aura-finetuned")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()

    train_path = Path(args.train_file)
    if not train_path.exists():
        alternative = Path("data/personality_training/aura.jsonl")
        if alternative.exists():
            train_path = alternative
        else:
            raise SystemExit(
                f"No training file found at {args.train_file} or {alternative}"
            )

    print(f"Loading training data from {train_path}")
    texts = [build_text(example) for example in load_jsonl(train_path)]
    texts = [text for text in texts if text]
    if not texts:
        raise SystemExit("No usable training examples found after filtering.")

    try:
        import datasets  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "Missing packages: install `transformers` and `datasets`. "
            f"Error: {exc}"
        ) from exc

    from core.runtime.model_lane_control import standalone_model_lane

    with standalone_model_lane(
        owner_id="gpt2-persona-finetune",
        model_path=args.model,
        purpose="train",
        preemptible=False,
        metadata={"tool": "scripts.run_finetune_gpt2"},
    ):
        _run_training(args, texts)


if __name__ == "__main__":
    main()
