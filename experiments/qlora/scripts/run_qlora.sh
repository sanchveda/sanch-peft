#!/usr/bin/env bash

set -euo pipefail

python -m core.training.train --config experiments/qlora/configs/base_qlora.yaml
