# Code Architecture

This document contains the detailed directory structure for `sanch-peft`. The repository is organized to keep reusable source code separate from experiment definitions, written notes, and project-level documentation.

## Directory Tree

```text
sanch-peft/
  README.md
  CODE_ARCHITECTURE.md
  .gitignore
  pyproject.toml
  requirements.txt

  core/
    __init__.py

    model/
      __init__.py

    training/
      __init__.py
      train.py
      evaluate.py
      config.py

    peft/
      __init__.py
      lora.py
      qlora.py

    utils/
      __init__.py
      logging_utils.py
      metrics.py

  experiments/
    baseline/
      configs/
        base_full_ft.yaml
      scripts/
        run_baseline.sh
      logs/
        .gitkeep
      results/
        .gitkeep

    lora/
      configs/
        base_lora.yaml
      scripts/
        run_lora.sh
      logs/
        .gitkeep
      results/
        .gitkeep

    qlora/
      configs/
        base_qlora.yaml
      scripts/
        run_qlora.sh
      logs/
        .gitkeep
      results/
        .gitkeep

  weeks/
    week05-peft/
      README.md
      notes.md
      results.md

  docs/
    roadmap.md
    repo_structure.md
```

## Structure Notes

### `core/`

Reusable Python source code. This is the implementation layer of the repository.

- `model/`: model-specific wrappers or loaders
- `training/`: training entrypoints, evaluation hooks, and config loading
- `peft/`: LoRA and QLoRA source code scaffolding
- `utils/`: shared logging and metric helpers

### `experiments/`

Runnable experiment definitions separated from reusable code.

- `baseline/`: baseline full fine-tuning comparison setup
- `lora/`: LoRA configs and launch scripts
- `qlora/`: QLoRA configs and launch scripts

Each experiment area follows the same layout:

- `configs/`: YAML experiment definitions
- `scripts/`: shell entrypoints
- `logs/`: run logs
- `results/`: summarized outputs and artifacts

### `weeks/`

Markdown-first study and reporting area.

- `week05-peft/README.md`: weekly scope
- `week05-peft/notes.md`: concept notes and open questions
- `week05-peft/results.md`: experiment writeups and comparisons

### `docs/`

Project-level documentation for planning and organization.

- `roadmap.md`: near-term and future direction
- `repo_structure.md`: concise explanation of repository areas

## Guiding Rules

- Keep reusable code under `core/`
- Keep experiment-specific configs and scripts under `experiments/`
- Keep notes and written findings under `weeks/`
- Keep planning and repository documentation under `docs/`
- Keep LoRA and QLoRA under `core/peft/`
- Avoid notebooks so the repository remains source-code-first and version-control-friendly
