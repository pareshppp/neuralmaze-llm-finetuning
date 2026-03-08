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
Hugging Face Job: Full Finetuning Experiment (SFT - Instruction Tuning)
--------------------------------------------------------------------------------
Author: The Neural Maze
Context: Lecture on LLM Finetuning Techniques.

DESCRIPTION:
This script performs Full Finetuning (Supervised Fine-Tuning) on a small
Language Model (SLM). Unlike LoRA/QLoRA (which only train adapter layers),
Full Finetuning updates 100% of the model's weights on instruction/response
pairs, making the model follow instructions and adopt a specific behavior or
conversational style.

KEY CONCEPTS DEMONSTRATED:
1. Full Finetuning: We are NOT using LoRA. We are updating 100% of the weights.
   This produces the highest-quality result but requires more VRAM and compute.
2. Chat Template Formatting: Input data is formatted using the model's chat
   template (apply_chat_template) so sequences match the model's expected input
   format for instruction following.
3. Unsloth Optimization: Using Unsloth to speed up the forward/backward passes
   even without quantization.

CLI INSTRUCTION
hf jobs uv run main.py `
    --flavor a10g-small `
    --timeout 3h `
    -e COMET_PROJECT_NAME="finetuning-sessions-lab2" `
    -s COMET_API_KEY="KiCQa4w2mlTNZFih4hAw76XZV" `
    -s HF_TOKEN="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"
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
formatter = log.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
root.addHandler(handler)


def main(
    # --- MODEL PARAMETERS ---
    model_name: str = "Qwen/Qwen3-0.6B", 
    # Qwen3-0.6B is excellent for demos: fast, capable, and fits in small VRAM.
    
    # --- DATASET PARAMETERS ---
    dataset_name: str = "theneuralmaze/finetuning-sessions-dataset",
    dataset_column: str = "messages_no_thinking",  # or "messages_thinking" to include <think> traces
    dataset_num_rows: int = None,  # None = full train split; set an int (e.g. 500) for a quick demo run
    eval_num_rows: int = None,  # None = full validation split

    # --- TRAINING PARAMETERS ---
    output_dir: str = "outputs",
    hub_model_id: str = "Qwen3-0.6B-Full-Finetuning",
    max_seq_length: int = 2048,
    batch_size: int = 4,
    gradient_accumulation_steps: int = 4,
    learning_rate: float = 2e-5, # Lower LR for full finetuning to prevent "catastrophic forgetting"
    num_train_epochs: int = 1,
    max_steps: int = -1,  # -1 = disabled; set a positive int (e.g. 100) to stop after N steps regardless of epochs
) -> None:
    
    log.info("================================================================")
    log.info("       STARTING FULL FINETUNING EXPERIMENT (SFT)               ")
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
        full_finetuning=True,
    )
    
    log.info("Model loaded successfully in full precision.")

    # --------------------------------------------------------------------------
    # STEP 2: PREPARE DATASET
    # --------------------------------------------------------------------------
    # Supervised Fine-Tuning requires instruction/response pairs, NOT raw text.
    # Each example contains a 'messages_no_thinking' field with a structured
    # conversation (user prompt + assistant response), which we format using the
    # model's chat template so the tokenizer sees the correct special tokens.
    # --------------------------------------------------------------------------
    train_split = "train" if dataset_num_rows is None else f"train[:{dataset_num_rows}]"
    eval_split  = "validation" if eval_num_rows is None else f"validation[:{eval_num_rows}]"
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
    eval_dataset  = load_dataset(dataset_name, split=eval_split).map(build_prompt)

    log.info(f"Train samples: {len(train_dataset)} | Eval samples: {len(eval_dataset)}")

    # --------------------------------------------------------------------------
    # STEP 3: CONFIGURE TRAINING
    # --------------------------------------------------------------------------
    # For SFT we do NOT use packing. Each example is a self-contained conversation
    # and we want the model to learn the boundary between the user prompt and the
    # assistant response independently — mixing multiple conversations into one
    # packed block would corrupt that signal.
    #
    # NOTE: The SFTTrainer handles masking of the prompt tokens automatically
    # so that loss is computed only on the assistant's response tokens, not on
    # the user input. This is the standard SFT training objective.
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
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        optim = "adamw_8bit", # Saves optimizer state memory, allowing larger batch sizes
        weight_decay = 0.01,
        
        # Evaluation
        eval_strategy="steps",
        eval_steps=100,

        # Logging & Saving
        logging_steps=2,
        save_strategy="no", # We only save at the end for this job
        report_to=["comet_ml"],
        seed=3407,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text", # The column name containing the raw text
        max_seq_length=max_seq_length,
        dataset_num_proc=min(8, os.cpu_count()-2),
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
    
    log.info(f"GPU: {gpu_stats.name} (Compute Capability: {gpu_stats.major}.{gpu_stats.minor})")
    log.info(f"Total VRAM Available: {gpu_stats.total_memory / 1024**3:.2f} GB")
    log.info(f"Pre-Train VRAM Used:  {start_gpu_memory:.2f} GB (Weights + Optim Loaded)")
    
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
    # Since we performed Full Finetuning (not LoRA), we save the ENTIRE model
    # (config + all weight tensors), not just a small adapter file.
    # This results in a larger upload (~1-2 GB for a 0.6B model) but the result
    # is a self-contained model that any HF-compatible framework can load directly.
    # The model is published under the theneuralmaze org on the Hugging Face Hub.
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