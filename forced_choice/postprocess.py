import argparse
import os
from collections import Counter

import jsonlines as jsonl

from forced_choice.restricted import constrained_surprisals
from forced_choice.variables import ANSWER_CHOICES

"""
Postprocessing for the output of forced_choice_script.py.

Adds to every line two versions of which answer choice the model preferred, each an integer
1-4 (1 = lowest, e.g. Very Unlikely, 4 = highest, e.g. Very Likely; the labels for each key
are stored in the file as {k}_choice):
  - choice: the choice with the lowest surprisal per token ({k}_mean_surprisal), i.e. over the
    full vocabulary, length-normalized.
  - choice_constrained: the choice with the lowest {k}_constrained_surprisal, i.e. with the
    output restricted to the four choices token by token (see forced_choice/restricted.py), so
    tokens that only complete a label cost nothing and no length normalization is needed.
    Files written before forced_choice_script.py added that field get it computed here from
    {k}_toks and {k}_surprisal (same calculation; the check that no choice merges with the end
    of the prompt needs the tokenizer, so it is skipped for those). None where the field is
    None (an answer choice merged with the end of the prompt), and for surprisal_script.py
    output, whose keys all score the same target tokens rather than different answers.

The choices differ in length ("Likely" is 2 tokens, "Very Unlikely" is 3), so the raw
sums are not comparable; dividing by token count puts them on the same scale. That is
already stored per line as {k}_mean_surprisal by forced_choice_script.py.

python -m forced_choice.postprocess --in-path data/forced_choice_output/Llama-3.1-8B-Instruct.jsonl --out-path data/forced_choice_processed/Llama-3.1-8B-Instruct.jsonl
"""

CHOICE_KEYS = [1, 2, 3, 4]


def choice_labels(row) -> dict:
    """Label per key, from the file ({k}_choice, written by forced_choice_script.py for its
    --scale). surprisal_script.py output has none, so fall back to the "likely" labels."""
    return {k: row.get(f"{k}_choice", ANSWER_CHOICES[k - 1]) for k in CHOICE_KEYS}


def lowest_key(scores: dict):
    """The key with the lowest score, and whether anything tied with it.

    min() picks the lowest-numbered choice if two ever tie exactly.
    """
    lowest = min(scores.values())
    return min(CHOICE_KEYS, key=lambda k: scores[k]), sum(1 for v in scores.values() if v == lowest) > 1


def add_choice(obj):
    """Add both versions of the preferred choice. Returns (obj, tied, tied_constrained)."""
    normalized = {}
    for k in CHOICE_KEYS:
        surprisal = obj[f"{k}_surprisal"]
        if not surprisal:
            raise ValueError(f"{obj.get('unique_id')} ({obj.get('condition')}): "
                             f"no surprisal values for choice {k}")
        normalized[k] = sum(surprisal) / len(surprisal)
    obj["choice"], tied = lowest_key(normalized)

    if len({tuple(obj[f"{k}_toks"]) for k in CHOICE_KEYS}) < len(CHOICE_KEYS):
        # not alternative answer strings, e.g. surprisal_script.py output, where every key
        # scores the same target after a different intervention: restriction is undefined
        obj["choice_constrained"] = None
        return obj, tied, False
    if any(f"{k}_constrained_surprisal" not in obj for k in CHOICE_KEYS):
        # older output file: compute it from the stored per-token values
        per_token = constrained_surprisals([obj[f"{k}_toks"] for k in CHOICE_KEYS],
                                           [obj[f"{k}_surprisal"] for k in CHOICE_KEYS])
        for k, values in zip(CHOICE_KEYS, per_token):
            obj[f"{k}_constrained_surprisal"] = sum(values)
            obj[f"{k}_constrained_token_surprisal"] = values
    constrained = {k: obj[f"{k}_constrained_surprisal"] for k in CHOICE_KEYS}
    tied_constrained = False
    if any(v is None for v in constrained.values()):
        obj["choice_constrained"] = None
    else:
        obj["choice_constrained"], tied_constrained = lowest_key(constrained)
    return obj, tied, tied_constrained


def print_distribution(counts, label, labels):
    total = sum(counts.values())
    print(f"{label}:")
    for k in CHOICE_KEYS:
        n = counts[k]
        print(f"  {k} {labels[k]:<16} {n:>6} ({n / total:.1%})" if total else f"  {k} {labels[k]}")


def run(in_path, out_path, force=False):
    if os.path.abspath(in_path) == os.path.abspath(out_path):
        raise ValueError("--out-path must differ from --in-path, this does not edit in place")
    if os.path.exists(out_path) and not force:
        raise ValueError(f"{out_path} already exists, pass --force to overwrite it")
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with jsonl.open(in_path, 'r') as reader:
        rows = list(reader)
    if not rows:
        raise ValueError(f"{in_path} is empty (is the run still going?)")

    counts, counts_constrained = Counter(), Counter()
    n_ties = n_ties_constrained = n_missing = 0
    with jsonl.open(out_path, 'w') as writer:
        for obj in rows:
            obj, tied, tied_constrained = add_choice(obj)
            n_ties += tied
            n_ties_constrained += tied_constrained
            counts[obj["choice"]] += 1
            if obj["choice_constrained"] is None:
                n_missing += 1
            else:
                counts_constrained[obj["choice_constrained"]] += 1
            writer.write(obj)

    print(f"\n{in_path} -> {out_path}")
    print(f"{len(rows)} lines")
    labels = choice_labels(rows[0])
    print_distribution(counts, "choice (surprisal per token)", labels)
    n_constrained = len(rows) - n_missing
    if n_constrained:
        print_distribution(counts_constrained, "choice_constrained (restricted to the four choices)", labels)
        agree = sum(1 for o in rows if o["choice_constrained"] is not None
                    and o["choice"] == o["choice_constrained"])
        print(f"the two agree on {agree}/{n_constrained} lines ({agree / n_constrained:.1%})")
    if n_missing:
        print(f"NOTE: choice_constrained is None on {n_missing}/{len(rows)} lines (the keys score "
              f"identical tokens, e.g. surprisal_script.py output, or an answer choice merged "
              f"with the end of the prompt)")
    if n_ties_constrained:
        print(f"WARNING: {n_ties_constrained} lines had a tie for lowest constrained surprisal, "
              f"resolved to the lower-numbered choice")
    if n_ties:
        print(f"WARNING: {n_ties} lines had a tie for lowest surprisal per token, "
              f"resolved to the lower-numbered choice")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Add the chosen answer to forced-choice output")
    ap.add_argument("--in-path", required=True, help="Raw output from forced_choice_script.py")
    ap.add_argument("--out-path", required=True)
    ap.add_argument("--force", action="store_true", help="Overwrite --out-path if it already exists")

    args = ap.parse_args()

    run(in_path=args.in_path,
        out_path=args.out_path,
        force=args.force)
