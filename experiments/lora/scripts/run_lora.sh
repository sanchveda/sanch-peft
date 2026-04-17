#!/usr/bin/env bash

set -euo pipefail

python -m core.training.train --config experiments/lora/configs/base_lora.yaml
