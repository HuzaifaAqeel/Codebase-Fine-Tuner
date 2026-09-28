# 🎯 Codebase Fine-Tuner

**Fine-tune a causal LLM on your own codebase with LoRA — teach the model your code style, idioms, and APIs.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE.md)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)

Generic code models know Python. They don't know *your* Python — your helper libraries, your naming
conventions, your internal APIs. This tool walks a codebase, builds a language-modelling dataset from
it, and LoRA fine-tunes a causal LLM on it, so completions start sounding like they were written by
your own team.

## How it works

1. **Dataset prep** — recursively walks `--input_dir`, honours `.codeignore` patterns, filters by
   `--file_extensions`, optionally strips comments, and concatenates everything into one training file
   with `<file_start>` / `<file_end>` markers.
2. **Tokenization** — loads the file with 🤗 `datasets`, splits 90/10 train/validation, tokenizes with
   the base model's tokenizer (labels = input_ids, causal LM).
3. **LoRA training** — freezes the base model, injects low-rank adapters (`peft`), and trains with
   🤗 `Trainer`: configurable rank/alpha/target modules, gradient accumulation, early stopping,
   optional 8-bit quantization and fp16.
4. **Before/after sampling** — generates a code sample from the base model, then from the tuned
   adapter, so you can see the difference fine-tuning made.

## Installation

```bash
pip install -r requirements.txt
```

Python 3.10+, PyTorch required. GPU recommended for real runs; the smoke demo runs on CPU.

## Quick smoke test (no GPU, ~2 minutes)

Trains a tiny random GPT-2 on a synthetic micro-corpus — proves the whole pipeline works end to end:

```bash
python fine_tune_codebase.py --smoke
```

You'll see the training loss decrease step by step, then a **before/after** generation sample:
the base model emits noise, the LoRA-tuned model reproduces the code pattern it learned.

## Full run

```bash
python fine_tune_codebase.py \
  --input_dir ./my_project \
  --file_extensions .py \
  --model_name codellama/CodeLlama-7b-hf \
  --output_dir ./fine_tuned_model \
  --batch_size 4 \
  --num_epochs 3 \
  --learning_rate 5e-5 \
  --lora_r 8 --lora_alpha 32 \
  --target_modules q_proj v_proj \
  --test_prompt "def load_config("
```

Use `--preprocess` to strip comments, `--fp16` for mixed precision (GPU),
`--quantize` for 8-bit base weights (GPU), `--early_stopping_patience 2` to stop
on validation plateau, and `--interactive` for a prompt loop after training.

## Options

| Flag | Default | Description |
|------|---------|-------------|
| `--input_dir` | — | Codebase to train on (required unless `--smoke`) |
| `--smoke` | off | Tiny-data mode: tiny model, synthetic corpus, few dozen steps |
| `--model_name` | `codellama/CodeLlama-7b-hf` | Base causal LM |
| `--output_dir` | `./fine_tuned_model` | Where the LoRA adapter is saved |
| `--file_extensions` | all | e.g. `--file_extensions .py .js` |
| `--max_steps` | -1 | Cap steps (overrides `--num_epochs` when > 0) |
| `--lora_r` / `--lora_alpha` | 8 / 32 | LoRA rank and scaling |
| `--target_modules` | `q_proj v_proj` | Modules to adapt (`c_attn` for GPT-2) |
| `--test_prompt` | `def add(a, b):` | Prompt for the before/after sample |

## Project structure

```
fine_tune_codebase.py   # the whole pipeline: prep -> tokenize -> LoRA train -> sample
.codeignore             # ignore patterns for dataset prep
requirements.txt
LICENSE.md
```

## Notes

- Set `HF_TOKEN` in your environment if you use gated models.
- Training checkpoints and the tokenizer are saved to `--output_dir`; load the
  adapter later with `peft.PeftModel.from_pretrained(base, output_dir)`.

## License

MIT — see [LICENSE.md](LICENSE.md).
