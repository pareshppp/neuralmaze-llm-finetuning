#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "fire>=0.7.1",
#   "comet_ml",
#   "unsloth",
#   "transformers",
#   "datasets>=4.4.0",
#   "trl",
#   "huggingface_hub"
# ]
# ///

"""
main.py
--------------------------------------------------------------------------------
Hugging Face Job: Continued Pretraining (CPT) with Full Finetuning
--------------------------------------------------------------------------------
Author: The Nerual Maze
Context: Lecture on LLM Domain Adaptation.

DESCRIPTION:
This script performs Continued Pretraining on a small Language Model (SLM).
Unlike Instruction Tuning (which uses Q&A pairs), CPT uses raw text to teach
the model a new domain (e.g., Law, Medicine, Finance).

KEY CONCEPTS DEMONSTRATED:
1. Full Finetuning: We are NOT using LoRA. We are updating 100% of the weights.
2. Packing: Concatenating short documents into full context blocks (2048 tokens)
   to maximize training efficiency and stabilize the loss.
3. Unsloth Optimization: Using Unsloth to speed up the forward/backward passes
   even without quantization.

CLI INSTRUCTION
hf jobs uv run main.py `
    --flavor a10g-small `
    --timeout 3h `
    -e COMET_PROJECT_NAME="neuralmaze-finetuning-lab1" `
    -s COMET_API_KEY=<your_comet_api_key> `
    -s HF_TOKEN=<your_hf_token> `
"""

import comet_ml
import sys
import logging as log
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
    model_name: str = "Qwen/Qwen3-0.6B-Base",
    # Qwen2.5-0.5B is excellent for demos: fast, capable, and fits in small VRAM.
    # --- DATASET PARAMETERS ---
    dataset_name: str = "pritamdeb68/Math-Pretraining-Data",
    # --- TRAINING PARAMETERS ---
    output_dir: str = "outputs",
    hub_model_id: str = "Qwen3-0.6B-Base-CPT-Math",
    max_seq_length: int = 1024,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-5,  # Lower LR for full finetuning to prevent "catastrophic forgetting"
    num_train_epochs: int = 1,
) -> None:

    log.info("================================================================")
    log.info("       STARTING CONTINUED PRETRAINING (FULL FINETUNE)           ")
    log.info("================================================================")

    # --------------------------------------------------------------------------
    # STEP 1: LOAD MODEL
    # --------------------------------------------------------------------------
    # For Full Finetuning, we explicitly disable 4-bit quantization (`load_in_4bit=False`).
    # We want the raw weights (bfloat16 or float32) because we are going to calculate
    # gradients for EVERY parameter in the network, not just adapters.
    # --------------------------------------------------------------------------
    log.info(f"Loading Base Model: {model_name}...")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,  # Auto-detects hardware capability (likely bfloat16 on A10G)
        load_in_4bit=False,
    )

    # Enable gradient checkpointing to save memory
    model.gradient_checkpointing_enable()

    log.info("Model loaded successfully in full precision.")

    # --------------------------------------------------------------------------
    # STEP 2: PREPARE DATASET
    # --------------------------------------------------------------------------
    # Continued Pretraining requires massive amounts of RAW text.
    # We use 'pritamdeb68/Math-Pretraining-Data' to inject math knowledge.
    # We stream or slice the dataset to keep this demo efficient.
    # --------------------------------------------------------------------------
    log.info(f"Loading Domain Dataset: {dataset_name}...")

    # Using the first 500 examples for this demo job to finish quickly.
    # In production, you would use the full 'train' split.
    dataset = load_dataset(dataset_name, split=f"train[:10000]")

    log.info(f"Dataset loaded. Samples: {len(dataset)}")

    # --------------------------------------------------------------------------
    # STEP 3: CONFIGURE TRAINING
    # --------------------------------------------------------------------------
    # We use packing=True. This concatenates short docs (A, B, C) into one long block.
    # This is efficient for Pretraining because we fill the entire context window.
    # That is the way CPT is meant to be.
    #
    # NOTE: Do not confuse this with `group_by_length=True`.
    # `group_by_length` sorts data so similar lengths are batched together (used in Chat/SFT).
    # Since packing makes ALL sequences the same length, grouping is redundant here.
    # This technique is more appropriate for text completion, where a conversational
    # structure of text is assumed via special tokens.
    # --------------------------------------------------------------------------
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        warmup_steps=100,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        # Optimization settings
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        optim="adamw_8bit",  # Saves optimizer state memory, allowing larger batch sizes
        weight_decay=0.01,
        gradient_checkpointing=True,  # Enabled via model.gradient_checkpointing_enable()
        # Logging & Saving
        logging_steps=20,
        save_strategy="no",  # We only save at the end for this job
        report_to=["comet_ml"],
        seed=3407,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",  # The column name containing the raw text
        max_seq_length=max_seq_length,
        dataset_num_proc=min(8, os.cpu_count() - 2),
        packing=True,  # <--- CRITICAL for Continued Pretraining
        args=training_args,
    )

    # --------------------------------------------------------------------------
    # STEP 4: EXECUTE TRAINING
    # --------------------------------------------------------------------------
    # When you call trainer.train(), the GPU memory fills up in 4 distinct "Buckets".
    #
    # BUCKET 1, 2, 3: FIXED COSTS (Static Memory)
    # -> Model Weights, Gradients, Optimizer States.
    # -> These depend ONLY on the model size (0.5B params).
    # -> They do NOT change if you increase Batch Size or Context Length.
    #
    # BUCKET 4: ACTIVATIONS (Variable Memory - The Dangerous One)
    # -> This bucket stores the math for the "Forward Pass".
    # -> Size = Batch Size × Context Length.
    #
    # CRITICAL NOTE ON CONTEXT LENGTH & FLASH ATTENTION:
    # In older transformers, Activations grew QUADRATICALLY (N^2) with context length.
    # Double the context = 4x the memory. This made long documents analysis impossible.
    #
    # Unsloth uses "Flash Attention 2", which makes scaling LINEAR (N).
    # Double the context = 2x the memory.
    #
    # HARDWARE REQUIREMENT:
    # To use Flash Attention, you MUST have a GPU with Compute Capability 8.0+
    # (Architecture: Ampere, Ada Lovelace, Hopper).
    # Examples: A100, A10G, H100, RTX 3090/4090.
    # Older GPUs (V100, T4) fall back to the slow, memory-hungry path.
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
        f"Pre-Train VRAM Used:  {start_gpu_memory:.2f} GB (Weights + Optim Loaded)"
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

    # LECTURE NOTE:
    # If 'Peak VRAM' is close to 'Total VRAM', you were near an OOM crash.
    # Solution: Lower 'per_device_train_batch_size' or 'max_seq_length'.

    # --------------------------------------------------------------------------
    # STEP 5: SAVE & PUSH TO HUB
    # --------------------------------------------------------------------------
    # Since we performed Full Finetuning, we are saving the WHOLE model (config + weights),
    # not just an adapter file. This results in a larger upload (approx 1-2GB for 0.5B model).
    # --------------------------------------------------------------------------
    log.info(f"Pushing model to Hugging Face Hub: {hub_model_id}...")

    # Push the tokenizer (vocabulary)
    tokenizer.push_to_hub(hub_model_id)

    # Push the model weights
    # Note: Unsloth's save_pretrained handles the conversion back to standard HF format
    model.push_to_hub(hub_model_id)

    log.info("Push complete! You can now load this model with:")
    log.info(f"Standard HF: AutoModelForCausalLM.from_pretrained('{hub_model_id}')")
    log.info(f"Unsloth: FastLanguageModel.from_pretrained('{hub_model_id}')")


if __name__ == "__main__":
    fire.Fire(main)
