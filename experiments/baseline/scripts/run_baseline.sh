#!/usr/bin/env bash

set -euo pipefail

python -m core.training.train --config experiments/baseline/configs/base_full_ft.yaml
