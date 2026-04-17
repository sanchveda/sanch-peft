# Repository Structure

## `core/`

Reusable source code for model logic, training entrypoints, PEFT implementations, and shared utilities. This directory should remain experiment-agnostic and form the stable code foundation of the lab.

## `experiments/`

Runnable experiment definitions, including YAML configs, launch scripts, logs, and result directories. This keeps one-off experimental setup separate from reusable implementation code.

## `weeks/`

Weekly notes, study artifacts, and result summaries. This is the written research layer of the repository and is useful for tracking learning progress and empirical findings over time.

## `docs/`

Project-level documentation such as roadmap notes and repository organization references. This directory explains intent and structure rather than storing experiment outputs.
