#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "fire>=0.7.1",
#   "comet_ml",
#   "unsloth",
#   "transformers",
#   "datasets",
#   "trl",
#   "huggingface_hub"
# ]
# ///

"""
main.py
--------------------------------------------------------------------------------
Hugging Face Job: LoRA Finetuning Experiment (SFT - Instruction Tuning)
--------------------------------------------------------------------------------
Author: The Neural Maze
Context: Lecture on LLM Finetuning Techniques.

DESCRIPTION:
This script performs LoRA (Low-Rank Adaptation) finetuning on a small
Language Model (SLM). Unlike Full Finetuning (which updates 100% of the weights),
LoRA injects small trainable matrices (adapters) into the model while keeping
the original weights frozen. This drastically reduces VRAM requirements and
training time while maintaining high performance.

KEY CONCEPTS DEMONSTRATED:
1. LoRA (Low-Rank Adaptation): We are NOT updating all weights. We only train
   low-rank matrices that "adapt" the base model.
2. Parameter-Efficient Fine-Tuning (PEFT): Only a tiny fraction (often <1%)
   of the model's parameters are updated.
3. Chat Template Formatting: Input data is formatted using the model's chat
   template (apply_chat_template) so sequences match the model's expected input
   format for instruction following.
4. Unsloth Optimization: Using Unsloth to speed up the forward/backward passes
   and efficiently manage LoRA adapters.

CLI INSTRUCTION
hf jobs uv run main.py `
    --flavor a10g-small `
    --timeout 3h `
    --max_steps 200 `
    -e COMET_PROJECT_NAME="finetuning-sessions-lab3" `
    -s COMET_API_KEY="XXXXXX" `
    -s HF_TOKEN="XXXXXX"
"""

import sys
import logging as log
import comet_ml
import torch
import fire
import os
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# --- LOGGING SETUP ---
# Configuring clear logs to track the job's progress in the HF console.
root = log.getLogger()
root.setLevel(log.INFO)
handler = log.StreamHandler(sys.stdout)
formatter = log.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
root.addHandler(handler)


def main(
    # --- MODEL PARAMETERS ---
    model_name: str = "Qwen/Qwen3-0.6B",
    # Qwen3-0.6B is excellent for demos: fast, capable, and fits in small VRAM.
    load_in_4bit: bool = False,  # Use 4-bit quantization to reduce VRAM usage
    # --- LORA PARAMETERS ---
    lora_r: int = 32,  # Rank of the LoRA matrices. Higher = more capacity, but more VRAM.
    lora_alpha: int = 16,  # Scaling factor for LoRA. Usually matches 'r'.
    lora_dropout: float = 0.0,  # Dropout for LoRA layers. 0 is standard for efficiency.
    # --- DATASET PARAMETERS ---
    dataset_name: str = "theneuralmaze/finetuning-sessions-dataset",
    dataset_column: str = "messages_thinking",  # "messages_no_thinking" or "messages_thinking" to include <think> traces
    dataset_num_rows: int = None,  # None = full train split; set an int (e.g. 500) for a quick demo run
    eval_num_rows: int = None,  # None = full validation split
    # --- TRAINING PARAMETERS ---
    output_dir: str = "outputs",
    hub_model_id: str = "Qwen3-0.6B-LoRA-Finetuning-neuralmaze",
    max_seq_length: int = 2048,  # Qwen3-0.6B supports up to 4096, but we use 2048 for faster training and lower VRAM.
    batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-4,  # Higher LR for LoRA compared to full finetuning
    num_train_epochs: int = 1,
    max_steps: int = -1,  # -1 = disabled; set a positive int (e.g. 100) to stop after N steps regardless of epochs
) -> None:

    log.info("================================================================")
    log.info("       STARTING LORA FINETUNING EXPERIMENT (SFT)               ")
    log.info("================================================================")

    # --------------------------------------------------------------------------
    # STEP 1: LOAD MODEL & ADD LORA ADAPTERS
    # --------------------------------------------------------------------------
    # For LoRA, we typically use 4-bit quantization (`load_in_4bit=True`).
    # We freeze the base model and only calculate gradients for the added
    # low-rank adapter matrices.
    # --------------------------------------------------------------------------
    log.info(f"Loading Base Model: {model_name}...")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,  # Auto-detects hardware capability
        load_in_4bit=load_in_4bit,
    )

    log.info("Adding LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",  # Unsloth's gradient checkpointing is optimized for LoRA training and can further reduce VRAM usage at the cost of some speed.
        random_state=3407,
        use_rslora=False,  # RSLoRA is a variant that adds random sign flipping to the LoRA updates for better generalization. We disable it here for simplicity.
        loftq_config=None,  # LoFTQ is a technique that combines LoRA with quantization-aware training. We disable it here for simplicity.
    )

    log.info("Model prepared for LoRA training.")

    # --------------------------------------------------------------------------
    # STEP 2: PREPARE DATASET
    # --------------------------------------------------------------------------
    # Supervised Fine-Tuning requires instruction/response pairs, NOT raw text.
    # Each example contains a 'messages_no_thinking' field with a structured
    # conversation (user prompt + assistant response), which we format using the
    # model's chat template so the tokenizer sees the correct special tokens.
    # --------------------------------------------------------------------------
    train_split = "train" if dataset_num_rows is None else f"train[:{dataset_num_rows}]"
    eval_split = (
        "validation" if eval_num_rows is None else f"validation[:{eval_num_rows}]"
    )
    log.info(f"Loading dataset: {dataset_name} | column={dataset_column}")
    log.info(f"  train split : {train_split}")
    log.info(f"  eval  split : {eval_split}")

    def build_prompt(row):
        prompt = tokenizer.apply_chat_template(
            row[dataset_column],
            tokenize=False,
            add_generation_prompt=False,  # since we already include an assistant message
        )
        return {"text": prompt}

    train_dataset = load_dataset(dataset_name, split=train_split).map(build_prompt)
    eval_dataset = load_dataset(dataset_name, split=eval_split).map(build_prompt)

    log.info(f"Train samples: {len(train_dataset)} | Eval samples: {len(eval_dataset)}")

    # --------------------------------------------------------------------------
    # STEP 3: CONFIGURE TRAINING
    # --------------------------------------------------------------------------
    # For SFT we do NOT use packing. Each example is a self-contained conversation
    # and we want the model to learn the boundary between the user prompt and the
    # assistant response independently.
    # --------------------------------------------------------------------------
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=100,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        max_steps=max_steps,
        # Optimization settings
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        optim="adamw_8bit",
        weight_decay=0.01,
        # Evaluation
        eval_strategy="steps",
        eval_steps=100,
        # Logging & Saving
        logging_steps=2,
        save_strategy="no",
        report_to=["comet_ml"],
        seed=3407,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=min(8, os.cpu_count() - 2),
        args=training_args,
    )

    # --------------------------------------------------------------------------
    # STEP 4: EXECUTE TRAINING
    # --------------------------------------------------------------------------
    # LoRA drastically reduces BUCKET 1, 2, 3 memory (Fixed Costs) because we
    # only store gradients and optimizer states for the tiny adapter matrices.
    # --------------------------------------------------------------------------

    log.info("Starting Training Loop...")

    # Check GPU memory before start to establish a baseline
    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = torch.cuda.memory_allocated() / 1024**3

    log.info(
        f"GPU: {gpu_stats.name} (Compute Capability: {gpu_stats.major}.{gpu_stats.minor})"
    )
    log.info(f"Total VRAM Available: {gpu_stats.total_memory / 1024**3:.2f} GB")
    log.info(
        f"Pre-Train VRAM Used:  {start_gpu_memory:.2f} GB (Base Model + LoRA Adapters)"
    )

    # 1. THE TRAINING LOOP STARTS
    trainer_stats = trainer.train()

    # 2. DIAGNOSTICS POST-TRAINING
    end_gpu_memory = torch.cuda.memory_allocated() / 1024**3
    peak_gpu_memory = torch.cuda.max_memory_allocated() / 1024**3

    log.info(f"Training Complete. Steps: {trainer_stats.global_step}")
    log.info(f"Final Training Loss: {trainer_stats.training_loss:.4f}")

    log.info("--- MEMORY DIAGNOSTICS ---")
    log.info(f"Peak VRAM Used: {peak_gpu_memory:.2f} GB (The 'High Water Mark')")
    log.info(f"End VRAM Used:  {end_gpu_memory:.2f} GB")

    # --------------------------------------------------------------------------
    # STEP 5: SAVE & PUSH TO HUB
    # --------------------------------------------------------------------------
    # For LoRA, we only save the ADAPTERS (small weight files) by default.
    # This makes the upload very fast and the repository very small (~50-100MB).
    # To use this model later, you load the base model and then attach these adapters.
    # --------------------------------------------------------------------------
    log.info(f"Pushing LoRA adapters to Hugging Face Hub: {hub_model_id}...")

    # Push the tokenizer and the LoRA adapters
    tokenizer.push_to_hub(hub_model_id)
    model.push_to_hub(hub_model_id)

    log.info("Push complete! You can now load this model with:")
    log.info(f"Unsloth: FastLanguageModel.from_pretrained('{hub_model_id}')")
    log.info("Note: This repository only contains the LoRA adapters.")


if __name__ == "__main__":
    fire.Fire(main)

