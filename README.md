# sanch-peft

## Overview

`sanch-peft` is a focused lab for parameter-efficient fine-tuning (PEFT) of large language models. The repository is built around clean, source-code-first implementations, reproducible experiments, and systematic analysis of memory, efficiency, and performance tradeoffs.

The current scope is intentionally narrow: LoRA and QLoRA only. That constraint keeps the codebase lightweight, readable, and practical for iterative engineering and research work.

This project avoids notebooks by design. All workflows live in version-controlled source files, shell scripts, YAML configs, and markdown writeups to make experiments easier to reproduce, compare, and maintain over time.

## Current Focus

- Baseline full fine-tuning as a comparison point
- LoRA from-scratch implementation scaffolding under `core/peft/`
- QLoRA preparation scaffolding for quantized PEFT workflows
- Reusable training and evaluation entrypoints
- Structured notes and result summaries for weekly study

## Repository Layout

```text
sanch-peft/
  core/           Reusable source code for training, PEFT, utilities, and model code
  experiments/    Runnable experiment configs, scripts, logs, and results
  weeks/          Weekly study notes, experiment observations, and result summaries
  docs/           Project roadmap and repository organization notes
```

Code under `core/` is intended to stay reusable and experiment-agnostic. Experiment-specific YAML files and launch scripts live under `experiments/`. The `weeks/` directory holds study notes and written result summaries rather than executable code.

See `CODE_ARCHITECTURE.md` for the full directory structure and file-level breakdown.

## How to Run

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the baseline starter experiment:

```bash
bash experiments/baseline/scripts/run_baseline.sh
```

Run the LoRA starter experiment:

```bash
bash experiments/lora/scripts/run_lora.sh
```

Run the QLoRA starter experiment:

```bash
bash experiments/qlora/scripts/run_qlora.sh
```

## Planned Extensions

- Expand LoRA coverage from scaffolding to end-to-end adapter injection
- Add quantized loading and preparation flows for QLoRA
- Introduce stronger evaluation and experiment tracking utilities
- Extend the lab to additional PEFT methods once LoRA and QLoRA are well understood
- Build systematic comparisons across memory footprint, trainable parameters, runtime, and task performance
