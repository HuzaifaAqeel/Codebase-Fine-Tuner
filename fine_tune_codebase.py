#!/usr/bin/env python3
"""Fine-tune a causal language model on a codebase using LoRA (peft).

Pipeline:
    1. prepare_dataset()  - walk a codebase dir (honouring .codeignore) and
                            concatenate the sources into one text file
    2. tokenize_dataset() - load the text file with HF datasets and tokenize it
    3. train_lora()       - LoRA fine-tune with Hugging Face Trainer
    4. generate_sample()  - sample code from the tuned model

All secrets (e.g. HF tokens) come from the environment (``HF_TOKEN``) - never
hardcode them. Set ``HF_TOKEN`` in your shell if you need gated models.

Quick smoke test (tiny random model, synthetic micro-corpus, ~1-3 min on CPU):
    python fine_tune_codebase.py --smoke

Full run:
    python fine_tune_codebase.py --input_dir ./my_project --file_extensions .py \\
        --model_name codellama/CodeLlama-7b-hf --output_dir ./fine_tuned_model
"""

from __future__ import annotations

import argparse
import fnmatch
import logging
import os
import re
from pathlib import Path

import numpy as np
from tqdm import tqdm

from datasets import Dataset, load_dataset, load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

SMOKE_MODEL = "hf-internal-testing/tiny-random-gpt2"
SMOKE_PROMPT = "def add(a, b):"

# A micro-corpus with a deliberately repetitive pattern so that even a tiny
# random model visibly memorises it within a couple hundred steps: the smoke
# demo shows training loss decreasing and generation going from noise to the
# learned code pattern.
SMOKE_SNIPPETS = [
    "def add(a, b):\n    return a + b\n",
    "def add(a, b):\n    result = a + b\n    return result\n",
    "def add(a, b):\n    # add two numbers\n    return a + b\n",
    "def add(a, b):\n    total = a + b\n    return total\n",
]


# --------------------------------------------------------------------------- #
# 1. Dataset preparation                                                      #
# --------------------------------------------------------------------------- #
def read_ignore_patterns(ignore_file: str) -> list[str]:
    """Read .codeignore-style patterns, one per line ('#' = comment)."""
    if not os.path.exists(ignore_file):
        return []
    with open(ignore_file, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]


def should_ignore(path: str, patterns: list[str]) -> bool:
    for pattern in patterns:
        if fnmatch.fnmatch(path, pattern) or fnmatch.fnmatch(os.path.basename(path), pattern):
            return True
    return False


def remove_comments(code: str) -> str:
    """Strip // and /* */ style comments."""
    code = re.sub(r"//.*?\n", "\n", code)
    code = re.sub(r"/\*.*?\*/", "", code, flags=re.DOTALL)
    return code


def prepare_dataset(
    input_dir: str,
    output_file: str,
    ignore_file: str = ".codeignore",
    file_extensions: list[str] | None = None,
    preprocess: bool = False,
) -> str:
    """Concatenate a codebase into a single text file for language modelling."""
    ignore_patterns = read_ignore_patterns(ignore_file)
    all_files: list[str] = []
    for root, dirs, files in os.walk(input_dir):
        dirs[:] = [d for d in dirs if not should_ignore(os.path.join(root, d), ignore_patterns)]
        for file in files:
            file_path = os.path.join(root, file)
            if should_ignore(file_path, ignore_patterns):
                continue
            if file_extensions and not any(file.endswith(ext) for ext in file_extensions):
                continue
            all_files.append(file_path)

    with open(output_file, "w", encoding="utf-8") as outfile:
        for file_path in tqdm(all_files, desc="Processing files"):
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as infile:
                    content = infile.read()
                    if preprocess:
                        content = remove_comments(content)
                    outfile.write(f"<file_start>{file_path}\n{content}\n<file_end>\n\n")
            except OSError as exc:
                logger.error("Error reading %s: %s", file_path, exc)
    logger.info("Dataset prepared: %d files -> %s", len(all_files), output_file)
    return output_file


def build_smoke_corpus(output_file: str, repeats: int = 40) -> str:
    """Write a tiny synthetic Python corpus for --smoke mode."""
    rng = np.random.RandomState(7)
    order = rng.permutation(repeats * len(SMOKE_SNIPPETS))
    with open(output_file, "w", encoding="utf-8") as f:
        for idx in order:
            f.write(SMOKE_SNIPPETS[idx % len(SMOKE_SNIPPETS)] + "\n")
    logger.info("Smoke corpus written to %s", output_file)
    return output_file


# --------------------------------------------------------------------------- #
# 2. Tokenization                                                             #
# --------------------------------------------------------------------------- #
def tokenize_dataset(
    output_file: str,
    model_name: str,
    max_length: int = 512,
    cache_dir: str = "tokenized_dataset",
    seed: int = 7,
):
    """Load the concatenated text file and tokenize it for causal LM."""
    if os.path.exists(cache_dir):
        logger.info("Loading tokenized dataset from %s ...", cache_dir)
        tokenized = load_from_disk(cache_dir)
    else:
        logger.info("Tokenizing dataset ...")
        dataset = load_dataset("text", data_files={"train": output_file})
        dataset = dataset["train"].train_test_split(test_size=0.1, seed=seed)

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            logger.warning("No pad_token - falling back to eos_token for padding.")
            tokenizer.pad_token = tokenizer.eos_token

        def tokenize_function(examples):
            toks = tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=max_length,
            )
            pad_id = tokenizer.pad_token_id
            toks["labels"] = [
                [-100 if tok == pad_id else tok for tok in ids]
                for ids in toks["input_ids"]
            ]
            return toks

        tokenized = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
        tokenized.save_to_disk(cache_dir)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenized, tokenizer


# --------------------------------------------------------------------------- #
# 3. LoRA training                                                            #
# --------------------------------------------------------------------------- #
def train_lora(
    tokenized_dataset,
    tokenizer,
    model_name: str,
    output_dir: str,
    learning_rate: float = 5e-5,
    batch_size: int = 4,
    num_epochs: int = 3,
    max_steps: int = -1,
    gradient_accumulation_steps: int = 1,
    fp16: bool = False,
    early_stopping_patience: int = 0,
    logging_steps: int = 10,
    eval_steps: int = 50,
    save_steps: int = 500,
    lora_r: int = 8,
    lora_alpha: int = 32,
    target_modules: list[str] | None = None,
    quantize: bool = False,
    seed: int = 7,
) -> tuple[Trainer, list[float]]:
    """Apply LoRA and fine-tune. Returns the trainer and per-step train losses."""
    logger.info("Loading base model '%s' ...", model_name)

    model_kwargs: dict = {}
    if quantize:
        from transformers import BitsAndBytesConfig

        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    if quantize:
        model = prepare_model_for_kbit_training(model)

    if model.config.pad_token_id is None or (
        tokenizer.pad_token_id is not None
        and model.config.pad_token_id != tokenizer.pad_token_id
    ):
        model.config.pad_token_id = tokenizer.pad_token_id

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules or ["q_proj", "v_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_epochs,
        max_steps=max_steps,
        learning_rate=learning_rate,
        logging_steps=logging_steps,
        eval_strategy="steps" if max_steps > 0 or num_epochs > 0 else "no",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=2,
        fp16=fp16,
        seed=seed,
        report_to="none",
        load_best_model_at_end=early_stopping_patience > 0,
    )

    callbacks = []
    if early_stopping_patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=early_stopping_patience))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        callbacks=callbacks,
    )

    logger.info("Starting LoRA fine-tuning ...")
    trainer.train()

    losses = [e["loss"] for e in trainer.state.log_history if "loss" in e]
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    logger.info("Fine-tuned LoRA model saved to %s", output_dir)
    return trainer, losses


# --------------------------------------------------------------------------- #
# 4. Generation                                                               #
# --------------------------------------------------------------------------- #
def generate_sample(model, tokenizer, prompt: str, max_new_tokens: int = 60) -> str:
    """Greedy generation of a code sample from a prompt."""
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    with __import__("torch").no_grad():
        out = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(out[0], skip_special_tokens=True)


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fine-tune a causal LLM on a codebase with LoRA.")
    p.add_argument("--input_dir", type=str, default=None,
                   help="Directory with the codebase (not needed with --smoke).")
    p.add_argument("--output_file", type=str, default="codebase.txt",
                   help="Concatenated dataset file.")
    p.add_argument("--model_name", type=str, default="codellama/CodeLlama-7b-hf",
                   help="Base model to fine-tune.")
    p.add_argument("--output_dir", type=str, default="./fine_tuned_model",
                   help="Where to save the LoRA adapter.")
    p.add_argument("--test_prompt", type=str, default=SMOKE_PROMPT,
                   help="Prompt used for the before/after generation sample.")
    p.add_argument("--ignore_file", type=str, default=".codeignore")
    p.add_argument("--file_extensions", type=str, nargs="+", default=None,
                   help="Only include these extensions, e.g. .py .js")
    p.add_argument("--learning_rate", type=float, default=5e-5)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_epochs", type=int, default=3)
    p.add_argument("--max_steps", type=int, default=-1,
                   help="Cap training steps (overrides epochs when > 0).")
    p.add_argument("--early_stopping_patience", type=int, default=0)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--fp16", action="store_true", help="Mixed precision (GPU only).")
    p.add_argument("--preprocess", action="store_true", help="Strip comments from sources.")
    p.add_argument("--quantize", action="store_true", help="8-bit base model (GPU).")
    p.add_argument("--lora_r", type=int, default=8)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--target_modules", type=str, nargs="+", default=None,
                   help="LoRA target modules, e.g. q_proj v_proj (GPT-2: c_attn).")
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--interactive", action="store_true", help="Prompt loop after training.")
    p.add_argument("--smoke", action="store_true",
                   help="Tiny-data mode: tiny random model + synthetic micro-corpus, "
                        "a few dozen steps - for tests and demos.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.smoke:
        if args.model_name == "codellama/CodeLlama-7b-hf":
            args.model_name = SMOKE_MODEL  # local dir path also accepted
        args.max_steps = 120 if args.max_steps <= 0 else args.max_steps
        args.num_epochs = 1000  # max_steps takes precedence
        args.learning_rate = 1e-3 if args.learning_rate == 5e-5 else args.learning_rate
        args.max_length = 128
        args.target_modules = args.target_modules or ["c_attn", "c_fc"]
        args.lora_r = 16 if args.lora_r == 8 else args.lora_r
        args.output_dir = args.output_dir if args.output_dir != "./fine_tuned_model" else "./smoke_model"
        build_smoke_corpus(args.output_file)
    elif not args.input_dir:
        raise SystemExit("--input_dir is required (or use --smoke for the tiny-data demo).")
    else:
        prepare_dataset(args.input_dir, args.output_file, args.ignore_file,
                        args.file_extensions, args.preprocess)

    tokenized_dataset, tokenizer = tokenize_dataset(
        args.output_file, args.model_name, max_length=args.max_length,
        cache_dir="tokenized_dataset_smoke" if args.smoke else "tokenized_dataset",
    )

    # BEFORE sample from the base model
    from transformers import AutoModelForCausalLM as _AM

    base_model = _AM.from_pretrained(args.model_name)
    before = generate_sample(base_model, tokenizer, args.test_prompt)
    print("\n===== BEFORE (base model) =====")
    print(before)
    del base_model

    trainer, losses = train_lora(
        tokenized_dataset, tokenizer,
        model_name=args.model_name,
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        max_steps=args.max_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        fp16=args.fp16,
        early_stopping_patience=args.early_stopping_patience,
        logging_steps=5 if args.smoke else 50,
        eval_steps=20 if args.smoke else 200,
        save_steps=10_000,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=args.target_modules,
        quantize=args.quantize,
    )

    if losses:
        print(f"\nTraining loss: first={losses[0]:.4f} last={losses[-1]:.4f} "
              f"(min={min(losses):.4f} over {len(losses)} logged steps)")

    # AFTER sample from the LoRA-tuned model (trainer.model is the PeftModel)
    after = generate_sample(trainer.model, tokenizer, args.test_prompt)
    print("\n===== AFTER (LoRA fine-tuned) =====")
    print(after)

    if args.interactive:
        print("\nInteractive mode - type a prompt, or 'exit' to quit.")
        model = PeftModel.from_pretrained(trainer.model, args.output_dir)
        while True:
            prompt = input(">>> ").strip()
            if prompt.lower() == "exit":
                break
            print(generate_sample(model, tokenizer, prompt))


if __name__ == "__main__":
    main()
