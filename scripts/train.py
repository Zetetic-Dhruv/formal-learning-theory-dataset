#!/usr/bin/env python3
"""
Fine-tune Qwen 2.5 on FLT dataset using QLoRA.

Usage:
    python scripts/train.py --config config/training_config.yaml

Requires: torch, transformers, peft, trl, bitsandbytes, datasets
"""

import argparse
import json
from pathlib import Path

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_chat_dataset(path: str) -> Dataset:
    """Load JSONL chat-format dataset."""
    examples = []
    with open(path) as f:
        for line in f:
            examples.append(json.loads(line))
    return Dataset.from_list(examples)


def format_chat(example: dict, tokenizer) -> dict:
    """Apply the model's chat template to messages."""
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/training_config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # --- Quantization ---
    quant_cfg = cfg.get("quantization", {})
    bnb_config = None
    if quant_cfg.get("load_in_4bit"):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=quant_cfg.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_compute_dtype=getattr(
                torch, quant_cfg.get("bnb_4bit_compute_dtype", "bfloat16")
            ),
            bnb_4bit_use_double_quant=quant_cfg.get(
                "bnb_4bit_use_double_quant", True
            ),
        )

    # --- Model ---
    model_cfg = cfg["model"]
    model_name = model_cfg["base_model"]
    print(f"Loading model: {model_name}")

    attn_impl = model_cfg.get("attn_implementation", "eager")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        torch_dtype=getattr(torch, model_cfg.get("torch_dtype", "bfloat16")),
        attn_implementation=attn_impl,
        device_map="auto",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- LoRA ---
    if bnb_config:
        model = prepare_model_for_kbit_training(model)

    lora_cfg = cfg["lora"]
    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        target_modules=lora_cfg["target_modules"],
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # --- Data ---
    data_cfg = cfg["data"]
    train_ds = load_chat_dataset(data_cfg["train_file"])
    eval_ds = None
    eval_path = data_cfg.get("eval_file")
    if eval_path and Path(eval_path).exists():
        eval_ds = load_chat_dataset(eval_path)

    # Apply chat template
    train_ds = train_ds.map(
        lambda ex: format_chat(ex, tokenizer), remove_columns=["messages"]
    )
    if eval_ds:
        eval_ds = eval_ds.map(
            lambda ex: format_chat(ex, tokenizer), remove_columns=["messages"]
        )

    # --- Training ---
    train_cfg = cfg["training"]
    training_args = TrainingArguments(
        output_dir=train_cfg["output_dir"],
        num_train_epochs=train_cfg["num_train_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        warmup_ratio=train_cfg["warmup_ratio"],
        weight_decay=train_cfg["weight_decay"],
        logging_steps=train_cfg["logging_steps"],
        save_strategy=train_cfg["save_strategy"],
        eval_strategy=train_cfg["eval_strategy"] if eval_ds else "no",
        bf16=train_cfg.get("bf16", True),
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
        dataloader_num_workers=train_cfg.get("dataloader_num_workers", 4),
        seed=train_cfg.get("seed", 42),
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        max_seq_length=train_cfg.get("max_seq_length", 2048),
    )

    print("Starting training...")
    trainer.train()

    # --- Save ---
    final_dir = Path(train_cfg["output_dir"]) / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    print(f"Model saved to {final_dir}")


if __name__ == "__main__":
    main()
