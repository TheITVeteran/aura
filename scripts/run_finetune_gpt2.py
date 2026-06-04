#!/usr/bin/env python3
"""Small, local fine-tune for persona using GPT-2 / DistilGPT2.
Uses `data/personality_training/starter_aura.jsonl` if present, otherwise falls back to
`data/personality_training/aura.jsonl`.

This is intended as a lightweight demo run on CPU. For production/large runs use
the `scripts/fine_tune_persona.sh` template with accelerate and a GPU.
"""
import argparse
import json
import os
from pathlib import Path

def load_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]

def build_text(example):
    # Prefer explicit assistant field, then generated, then response.
    user = example.get('user') or example.get('prompt') or ''
    assistant = example.get('assistant') or example.get('generated') or example.get('response') or ''
    # If assistant contains an unawaited coroutine repr, exclude it from training.
    if isinstance(assistant, str) and assistant.startswith("<coroutine"):
        assistant = ''
    return (user + "\n" + assistant).strip()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', default='data/personality_training/starter_aura.jsonl')
    parser.add_argument('--model', default='distilgpt2')
    parser.add_argument('--output-dir', default='outputs/aura-finetuned')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=2)
    args = parser.parse_args()

    train_path = Path(args.train_file)
    if not train_path.exists():
        alt = Path('data/personality_training/aura.jsonl')
        if alt.exists():
            train_path = alt
        else:
            raise SystemExit(f"No training file found at {args.train_file} or {alt}")

    print(f"Loading training data from {train_path}")
    examples = load_jsonl(train_path)
    texts = [build_text(e) for e in examples]
    texts = [t for t in texts if t]
    if not texts:
        raise SystemExit("No usable training examples found after filtering.")

    try:
        from datasets import Dataset
        from transformers import (
            AutoTokenizer,
            AutoModelForCausalLM,
            DataCollatorForLanguageModeling,
            TrainingArguments,
            Trainer,
        )
    except ImportError as e:
        raise SystemExit("Missing packages: install `transformers` and `datasets`. Error: " + str(e))

    ds = Dataset.from_dict({'text': texts})

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_fn(ex):
        return tokenizer(ex['text'], truncation=True, max_length=512)

    tokenized = ds.map(tokenize_fn, batched=True, remove_columns=['text'])

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

if __name__ == '__main__':
    main()
