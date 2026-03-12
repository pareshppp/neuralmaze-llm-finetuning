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
Hugging Face Job: QLoRA Finetuning Experiment (SFT - Instruction Tuning)
--------------------------------------------------------------------------------
Author: The Neural Maze
Context: Lecture on LLM Finetuning Techniques.

DESCRIPTION:
This script performs QLoRA (Quantized Low-Rank Adaptation) finetuning on a
small Language Model (SLM). QLoRA is an evolution of LoRA that further reduces
memory usage by quantizing the base model to 4-bit while maintaining 16-bit
precision for the adapter weights.

QLORA VS LORA:
- LoRA (Low-Rank Adaptation):
    * Pros: Faster training (no dequantization overhead), slightly more stable.
    * Cons: Higher VRAM usage (requires FP16/BF16 base model weights).
- QLoRA (Quantized LoRA):
    * Pros: Drastically lower VRAM (can fit much larger models on consumer GPUs).
      Uses 4-bit NormalFloat (NF4) and Double Quantization.
    * Cons: Slightly slower due to constant quantization/dequantization steps
      during the forward and backward passes.

KEY CONCEPTS DEMONSTRATED:
1. QLoRA (Quantized LoRA): We load the base model in 4-bit NF4 format to save
   significant VRAM.
2. Parameter-Efficient Fine-Tuning (PEFT): Only a tiny fraction of the model's
   parameters (the LoRA adapters) are updated.
3. Chat Template Formatting: Input data is formatted using the model's chat
   template so sequences match the model's expected input format.
4. Model Merging: Instead of saving only adapters, we merge them back into
   the base model to create a standalone 16-bit "full" model.

CLI INSTRUCTION
hf jobs uv run main.py `
    --flavor a10g-small `
    --timeout 3h `
    --max_steps 2000 `
    --num_train_epochs 1 `
    -e COMET_PROJECT_NAME="finetuning-sessions-lab4" `
    -s COMET_API_KEY="<comet_api_key>" `
    -s HF_TOKEN="<hf_token>"
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
    load_in_4bit: bool = True,  # QLoRA REQUIRES 4-bit quantization
    # --- LORA PARAMETERS ---
    lora_r: int = 32,  # Rank of the LoRA matrices.
    lora_alpha: int = 16,  # Scaling factor for LoRA.
    lora_dropout: float = 0.0,  # Dropout for LoRA layers.
    # --- DATASET PARAMETERS ---
    dataset_name: str = "theneuralmaze/finetuning-sessions-dataset",
    dataset_column: str = "messages_no_thinking",  # messages_thinking
    dataset_num_rows: int = None,
    eval_num_rows: int = None,
    # --- TRAINING PARAMETERS ---
    output_dir: str = "outputs",
    hub_model_id: str = "Qwen3-0.6B-QLoRA-Finetuning-neuralmaze-thinking",
    max_seq_length: int = 2048,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-4,
    num_train_epochs: int = 1,
    max_steps: int = -1,
) -> None:

    log.info("================================================================")
    log.info("       STARTING QLORA FINETUNING EXPERIMENT (SFT)              ")
    log.info("================================================================")

    # --------------------------------------------------------------------------
    # STEP 1: LOAD MODEL & ADD LORA ADAPTERS (QLORA)
    # --------------------------------------------------------------------------
    # QLoRA = 4-bit Quantization + LoRA Adapters.
    # By loading in 4-bit, we reduce the base model footprint by ~4x.
    # --------------------------------------------------------------------------
    log.info(f"Loading Base Model in 4-bit (QLoRA): {model_name}...")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,  # Auto-detects hardware capability (BF16 for A100/H100/A10G)
        load_in_4bit=load_in_4bit,
        # Note: Unsloth handles bnb_4bit_use_double_quant, bnb_4bit_quant_type (nf4),
        # and bnb_4bit_compute_dtype automatically when load_in_4bit=True.
        # Passing them as direct keyword arguments causes a TypeError.
    )

    log.info("Adding LoRA adapters to the quantized model...")
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
        use_gradient_checkpointing="unsloth",  # Reduces VRAM by re-calculating activations
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    log.info("Model prepared for QLoRA training.")

    # --------------------------------------------------------------------------
    # STEP 2: PREPARE DATASET
    # --------------------------------------------------------------------------
    train_split = "train" if dataset_num_rows is None else f"train[:{dataset_num_rows}]"
    eval_split = (
        "validation" if eval_num_rows is None else f"validation[:{eval_num_rows}]"
    )
    log.info(f"Loading dataset: {dataset_name} | column={dataset_column}")

    def build_prompt(row):
        prompt = tokenizer.apply_chat_template(
            row[dataset_column],
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": prompt}

    train_dataset = load_dataset(dataset_name, split=train_split).map(build_prompt)
    eval_dataset = load_dataset(dataset_name, split=eval_split).map(build_prompt)

    log.info(f"Train samples: {len(train_dataset)} | Eval samples: {len(eval_dataset)}")

    # --------------------------------------------------------------------------
    # STEP 3: CONFIGURE TRAINING
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
        # Paged AdamW is often used in QLoRA to handle OOM spikes by offloading to CPU RAM.
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
    log.info("Starting Training Loop...")

    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = torch.cuda.memory_allocated() / 1024**3

    log.info(f"GPU: {gpu_stats.name}")
    log.info(f"Total VRAM Available: {gpu_stats.total_memory / 1024**3:.2f} GB")
    log.info(f"Pre-Train VRAM Used (4-bit Base + Adapters): {start_gpu_memory:.2f} GB")

    trainer_stats = trainer.train()

    end_gpu_memory = torch.cuda.memory_allocated() / 1024**3
    peak_gpu_memory = torch.cuda.max_memory_allocated() / 1024**3

    log.info(f"Training Complete. Final Loss: {trainer_stats.training_loss:.4f}")
    log.info(f"Peak VRAM Used: {peak_gpu_memory:.2f} GB")

    # --------------------------------------------------------------------------
    # STEP 5: SAVE & PUSH FULL MODEL (MERGED)
    # --------------------------------------------------------------------------
    # In QLoRA, we usually save just adapters. However, to deploy easily,
    # we can merge the adapters back into the base model weights.
    # We save as 'merged_16bit' to ensure compatibility with all HF tools.
    # --------------------------------------------------------------------------
    log.info(f"Merging and pushing FULL model to Hugging Face Hub: {hub_model_id}...")

    # Save the merged model in 16-bit precision
    model.push_to_hub_merged(
        hub_model_id,
        tokenizer,
        save_method="merged_16bit",
        # token = os.environ.get("HF_TOKEN"), # Unsloth picks this up automatically from env
    )

    log.info("Push complete! The FULL merged model is now available on the Hub.")
    log.info(
        f"You can load it normally with: AutoModelForCausalLM.from_pretrained('{hub_model_id}')"
    )


if __name__ == "__main__":
    fire.Fire(main)
