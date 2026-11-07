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

# bash scripts/run_kaggle.sh equiv-scale: rerun only the DDP equivalence run
# with a lower GradScaler init scale (features and earlier results must exist).
# bash scripts/run_kaggle.sh diagnose: per-rank inf/grad/replica logging, fp16 and fp32.
if [ "${1:-}" = "diagnose" ]; then
  stamp "diagnose: c fp16"
  prep_env
  torchrun --nproc_per_node=2 -m wge.cli ddp-equiv --diagnose
  stamp "diagnose: b fp16"
  python -m wge.cli equiv --run b --diagnose
  stamp "diagnose: b fp32"
  python -m wge.cli equiv --run b --diagnose --fp32
  stamp "diagnose: c fp32"
  torchrun --nproc_per_node=2 -m wge.cli ddp-equiv --diagnose --fp32
  python -m wge.cli diagnose-report
  stamp "done in $(( $(date +%s) - START )) s"
  exit 0
fi

if [ "${1:-}" = "equiv-scale" ]; then
  stamp "equiv-scale: DDP 2x8, GradScaler init scale 32768"
  prep_env
  torchrun --nproc_per_node=2 -m wge.cli ddp-equiv --init-scale 32768
  python -m wge.cli report
  stamp "done in $(( $(date +%s) - START )) s"
  exit 0
fi

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
