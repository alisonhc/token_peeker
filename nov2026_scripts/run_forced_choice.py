"""Run forced choice on every instruct input file with one model, then postprocess each output.

Loads the model once and, for every *.jsonl in --in-dir except the *_forsurprisal* (base-model)
files:
  1. scores the four answer choices with forced_choice_script.py (chat template / model encoder,
     thinking off), offering the answer options of the scale in the file name
     (mainexp_excl_input_often.jsonl -> often: Very Rarely, Rarely, Often, Very Often),
  2. postprocesses the output to add choice and choice_constrained.

Layout of --out-dir:
  raw/<input name>.jsonl          output of forced_choice_script.py
  processed/<input name>.jsonl    the same lines with choice / choice_constrained added
  run_config.json                 arguments and library versions of the last run

A file whose raw output is already complete (same number of lines as the input) is not run again,
so the script can be restarted after a crash. A partial raw output is an error unless
--overwrite is passed (forced_choice_script.py appends, so rerunning it would duplicate lines).

Run from the repo root, e.g. via nov2026_scripts/run_forced_choice_deepseek_v4_flash.sh, or:
    python -m nov2026_scripts.run_forced_choice --model-name /models/DeepSeek-V4-Flash-0731 \
        --backend vllm --tensor-parallel-size 4 --batch-size 500 \
        --out-dir data/output/forced_choice_DeepSeek-V4-Flash-0731
"""

import argparse
import datetime
import glob
import json
import os
import sys

from forced_choice import postprocess
from forced_choice.forced_choice_script import run_inference
from forced_choice.variables import SCALES, SYSTEM_PROMPT
from tools.load_model import add_model_args, load_model


def count_lines(path: str) -> int:
    with open(path, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def scale_from_name(path: str) -> str:
    scale = os.path.splitext(os.path.basename(path))[0].rsplit("_", 1)[-1]
    if scale not in SCALES:
        raise ValueError(f"Cannot tell the scale of {path}: its name should end in "
                         f"_{{{','.join(SCALES)}}}.jsonl")
    return scale


def find_inputs(in_dir: str, only: list[str] | None) -> list[str]:
    paths = sorted(p for p in glob.glob(os.path.join(in_dir, "*.jsonl")) if "forsurprisal" not in p)
    if only:
        paths = [p for p in paths if os.path.basename(p) in only or os.path.splitext(os.path.basename(p))[0] in only]
        missing = set(only) - {os.path.basename(p) for p in paths} - {os.path.splitext(os.path.basename(p))[0] for p in paths}
        if missing:
            raise ValueError(f"--files not found in {in_dir}: {sorted(missing)}")
    if not paths:
        raise ValueError(f"No instruct input files in {in_dir}")
    return paths


def library_versions() -> dict:
    versions = {"python": sys.version.split()[0]}
    for name in ("torch", "transformers", "vllm"):
        try:
            versions[name] = __import__(name).__version__
        except Exception:
            pass
    return versions


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_args(ap)
    ap.add_argument("--in-dir", default="data/input/for_llm_inference")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--files", nargs="*", default=None,
                    help="Only these input files (names, with or without .jsonl)")
    ap.add_argument("--batch-size", type=int, default=500)
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--system-prompt", default=SYSTEM_PROMPT)
    ap.add_argument("--enable-thinking", action="store_true",
                    help="Leave the thinking block open (default: thinking off)")
    ap.add_argument("--overwrite", action="store_true",
                    help="Delete existing raw outputs and run every file again")
    args = ap.parse_args()

    inputs = find_inputs(args.in_dir, args.files)
    jobs = [(p, scale_from_name(p)) for p in inputs]  # check all names before loading the model
    raw_dir = os.path.join(args.out_dir, "raw")
    processed_dir = os.path.join(args.out_dir, "processed")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)

    todo = []
    for in_path, scale in jobs:
        raw_path = os.path.join(raw_dir, os.path.basename(in_path))
        n_in = count_lines(in_path)
        n_out = count_lines(raw_path) if os.path.exists(raw_path) else 0
        if n_out and args.overwrite:
            os.remove(raw_path)
            n_out = 0
        if n_out == n_in:
            print(f"done already: {raw_path} ({n_out} lines)")
        elif n_out:
            raise ValueError(f"{raw_path} has {n_out} of {n_in} lines (a crashed run?). Pass "
                             "--overwrite to start again, or delete that file")
        else:
            todo.append((in_path, scale, raw_path))
    print(f"{len(todo)} of {len(jobs)} files to run: {[os.path.basename(p) for p, _, _ in todo]}")

    config = {k: v for k, v in vars(args).items()}
    config.update(started=datetime.datetime.now().isoformat(timespec="seconds"),
                  versions=library_versions(), files={os.path.basename(p): s for p, s in jobs})
    with open(os.path.join(args.out_dir, "run_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    if todo:
        model = load_model(args)
        for in_path, scale, raw_path in todo:
            print(f"\n=== {os.path.basename(in_path)} (scale: {scale}) -> {raw_path}", flush=True)
            run_inference(in_path=in_path, out_path=raw_path, mol=model,
                          batch_size=args.batch_size, max_length=args.max_length,
                          chat_template=True, system_prompt=args.system_prompt,
                          enable_thinking=args.enable_thinking, scale=scale)

    for in_path, _ in jobs:
        name = os.path.basename(in_path)
        postprocess.run(in_path=os.path.join(raw_dir, name),
                        out_path=os.path.join(processed_dir, name), force=True)


if __name__ == "__main__":
    main()
