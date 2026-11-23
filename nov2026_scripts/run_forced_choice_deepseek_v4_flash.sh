#!/usr/bin/env bash
# Forced choice for DeepSeek-V4-Flash-0731 on every instruct input file in
# data/input/for_llm_inference (all *.jsonl except *_forsurprisal*), then postprocessing.
#
# Output: data/output/forced_choice_DeepSeek-V4-Flash-0731/{raw,processed}/<input name>.jsonl,
# plus run_config.json and run.log. Rerunning skips files whose raw output is complete.
#
# Needs Hopper (H100/H200) or Blackwell GPUs: vLLM's DeepSeek-V4 kernels do not run on A100.
# The FP8 checkpoint is 173 GB, so 4 x 80 GB GPUs (TP=4) is the minimum.
#
# Usage (from anywhere, with the vLLM env active or PYTHON set):
#   bash nov2026_scripts/run_forced_choice_deepseek_v4_flash.sh
#   TP=8 bash nov2026_scripts/run_forced_choice_deepseek_v4_flash.sh
#   bash nov2026_scripts/run_forced_choice_deepseek_v4_flash.sh --files normexp_input_likely   # one file
# Extra arguments are passed on to nov2026_scripts/run_forced_choice.py.
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON=${PYTHON:-python}
MODEL=${MODEL:-/models/DeepSeek-V4-Flash-0731}
TP=${TP:-4}
BATCH_SIZE=${BATCH_SIZE:-500}
OUT_DIR=${OUT_DIR:-data/output/forced_choice_DeepSeek-V4-Flash-0731}

mkdir -p "$OUT_DIR"
echo "=== $(date -Is) $MODEL TP=$TP -> $OUT_DIR" | tee -a "$OUT_DIR/run.log"

"$PYTHON" -m nov2026_scripts.run_forced_choice \
    --model-name "$MODEL" \
    --backend vllm \
    --tensor-parallel-size "$TP" \
    --batch-size "$BATCH_SIZE" \
    --out-dir "$OUT_DIR" \
    "$@" 2>&1 | tee -a "$OUT_DIR/run.log"
