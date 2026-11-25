#!/usr/bin/env bash
# DeepSeek-V4-Flash-0731 (FP8, 173 GB). Needs Hopper (H100/H200) or Blackwell GPUs: vLLM's
# DeepSeek-V4 kernels do not run on A100. 4 x 80 GB GPUs (TP=4) is the minimum; TP=8 to use 8.
# No Jinja chat template: the script uses the model's own encoding/encoding_dsv4.py in "chat"
# mode, so the prompt ends with "<｜Assistant｜></think>" (thinking off).
#
# Forced choice on every instruct input file, then postprocessing; see forced_choice_model.sh.
# Output: data/output/forced_choice_DeepSeek-V4-Flash-0731/
#
# Usage (from anywhere):
#   bash nov2026_scripts/run_forced_choice_deepseek_v4_flash.sh
#   bash nov2026_scripts/run_forced_choice_deepseek_v4_flash.sh --files normexp_input_likely   # one file
set -euo pipefail
export MODEL=${MODEL:-/models/DeepSeek-V4-Flash-0731}
export NAME=${NAME:-DeepSeek-V4-Flash-0731}
export TP=${TP:-4}
export MODEL_ARGS=${MODEL_ARGS:-""}
exec bash "$(dirname "$0")/forced_choice_model.sh" "$@"
