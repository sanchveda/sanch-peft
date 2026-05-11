#!/usr/bin/env bash
set -euo pipefail

# Format + lint everything in current repo.
# Run from the repo root: ./format.sh
ruff check . --fix
ruff format .
