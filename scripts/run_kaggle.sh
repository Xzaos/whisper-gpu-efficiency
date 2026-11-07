#!/usr/bin/env bash
# Whole experiment on a Kaggle GPU T4 x2 machine, about 20 minutes.
# Needs HF_TOKEN in the environment (the Kaggle cell sets it from the secret).
set -euo pipefail
cd "$(dirname "$0")/.."

stamp() { echo "[$(date -u +%H:%M:%S)] $*"; }
prep_env() {
  pip uninstall -y torchao >/dev/null 2>&1 || true
  export USE_TF=0 TRANSFORMERS_NO_TF=1
}
mkdir -p results/raw
START=$(date +%s)

stamp "stage 0: install"
prep_env
pip install -q "datasets<4" "transformers>=4.46,<5" peft soundfile
pip install -q -e .

stamp "stage 1: prepare 400 FLEURS hi_in clips and precompute features"
prep_env
if [ -f data/features.pt ]; then
  stamp "data/features.pt exists, skipping"
else
  python -m wge.cli prepare --n 400 --out data/features.pt
fi

stamp "stage 2: configs 1-4 and max batch sweep (single GPU)"
prep_env
python -m wge.cli bench

stamp "stage 3: single-GPU baseline for DDP (lora fp16, batch 16 or 8 x accum 2)"
prep_env
python -m wge.cli baseline

stamp "stage 4: DDP on 2 GPUs (lora fp16, 8 per GPU)"
prep_env
torchrun --nproc_per_node=2 -m wge.cli ddp-bench

stamp "stage 5: equivalence check, 30 steps each"
prep_env
python -m wge.cli equiv --run a
python -m wge.cli equiv --run b
python -m wge.cli equiv --run b_repeat
torchrun --nproc_per_node=2 -m wge.cli ddp-equiv

stamp "stage 6: report"
python -m wge.cli report
stamp "done in $(( $(date +%s) - START )) s"
