"""Explicit CUDA self-training utility with process-wide model ownership."""

from __future__ import annotations

from pathlib import Path
from typing import Any

MAX_SEQ_LENGTH = 2048
DTYPE = None
LOAD_IN_4BIT = True
TRAINING_DATA_FILE = Path("autonomy_engine/memory/training_data.jsonl")
OUTPUT_DIR = "autonomy_engine/brain/outputs"
MODEL_NAME = "unsloth/llama-3-8b-Instruct-bnb-4bit"


def _run_training(
    *,
    torch: Any,
    fast_language_model: Any,
    sft_trainer: Any,
    training_arguments: Any,
    load_dataset: Any,
) -> None:
    print("Loading Base Model (Llama-3-8b-Instruct)...")
    model, tokenizer = fast_language_model.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=DTYPE,
        load_in_4bit=LOAD_IN_4BIT,
    )
    model = fast_language_model.get_peft_model(
        model,
        r=16,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )
    dataset = load_dataset("json", data_files=str(TRAINING_DATA_FILE), split="train")

    print(f"Training on {len(dataset)} examples...")
    trainer = sft_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="output",
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_num_proc=2,
        args=training_arguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            max_steps=60,
            learning_rate=2e-4,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=1,
            output_dir=OUTPUT_DIR,
            optim="adamw_8bit",
        ),
    )
    trainer.train()
    model.save_pretrained("autonomy_engine/brain/evolved_v1")


def train_self() -> None:
    """Fine-tune the local Llama model after all preconditions pass."""
    print(">>> INITIATING CEREBRAL UPDATE (Fine-Tuning) <<<")
    try:
        import torch
        from datasets import load_dataset
        from transformers import TrainingArguments
        from trl import SFTTrainer
        from unsloth import FastLanguageModel
    except ImportError as exc:
        print(f"WARNING: Unsloth/Transformers not installed or import failed ({exc}).")
        print("Skipping actual training (Simulation Mode).")
        print(">>> CEREBRAL UPDATE COMPLETE (SIMULATED). <<<")
        return

    if not torch.cuda.is_available():
        print("WARNING: No NVIDIA GPU detected. Unsloth requires CUDA.")
        print("Skipping actual training (Simulation Mode).")
        print(">>> CEREBRAL UPDATE COMPLETE (SIMULATED). <<<")
        return
    if not TRAINING_DATA_FILE.exists():
        print(f"No training data found at {TRAINING_DATA_FILE}.")
        return

    from core.runtime.model_lane_control import standalone_model_lane

    with standalone_model_lane(
        owner_id="cuda-self-update",
        model_path=MODEL_NAME,
        purpose="train",
        preemptible=False,
        metadata={"tool": "scripts.self_update"},
    ):
        _run_training(
            torch=torch,
            fast_language_model=FastLanguageModel,
            sft_trainer=SFTTrainer,
            training_arguments=TrainingArguments,
            load_dataset=load_dataset,
        )
    print(">>> CEREBRAL UPDATE COMPLETE. NEW SYNAPSES FORMED. <<<")


if __name__ == "__main__":
    train_self()
