import argparse
import jsonlines as jsonl

from tools.model_of_language import *
from tools.load_model import add_model_args, exit_cleanly, load_model
from tools.variables import SCALES, DEFAULT_SCALE, get_interventions, get_intervention_dict

"""

Add --backend vllm (plus e.g. --tensor-parallel-size 4 --batch-size 1000) to score with vLLM instead
of transformers. To compare the two on the same weights, run the HF backend with --no-load-in-8bit.

Use --scale {likely,typical,probable,often} to choose the intervention scale (default: likely), e.g.
nohup python -m surprisal_script --model-name Qwen/Qwen3-8B-Base --scale typical --in-path data/input/for_llm_inference/mainexp_input_forbasemodels.jsonl --out-path data/output/mainexp_qwen3_8b_base_typical.jsonl

nohup python -m surprisal_script --model-name /public/hf/models/meta-llama/Meta-Llama-3.1-8B --model-nickname Llama3.1-8B --in-path data/input/450_llm_input_forsurprisal.jsonl --out-path data/output/450_llama3.1_8b_output.jsonl > llama_8b_450.out &!

nohup python -m surprisal_script --model-name /public/hf/models/google/gemma-3-12b-pt --model-nickname gemma-3-12b --in-path data/input/450_llm_input_forsurprisal.jsonl --out-path data/output/450_gemma3_12b_output.jsonl

nohup python -m surprisal_script --model-name /public/hf/models/google/gemma-3-27b-pt --model-nickname gemma-3-27b --in-path data/input/450_llm_input_forsurprisal.jsonl --out-path data/output/450_gemma3_27b_output.jsonl

nohup python -m surprisal_script --model-name allenai/Olmo-3-1025-7B --in-path data/input/450_llm_input_forsurprisal.jsonl --out-path data/output/450_olmo3_7b_output.jsonl

nohup python -m surprisal_script --model-name allenai/Olmo-3-1125-32B --in-path data/input/450_llm_input_forsurprisal.jsonl --out-path data/output/450_olmo3_32b_output.jsonl

nohup python -m surprisal_script --model-name /public/hf/models/meta-llama/Meta-Llama-3.1-70B --in-path data/input/450_llm_input_forsurprisal.jsonl --out-path data/output/450_llama3.1_70b_output.jsonl

nohup python -m surprisal_script --model-name Qwen/Qwen3-8B-Base --in-path data/input/450_llm_input_forsurprisal.jsonl --out-path data/output/450_qwen3_8b_base_output.jsonl

nohup python -m surprisal_script --model-name Qwen/Qwen3-30B-A3B-Base --in-path data/input/450_llm_input_forsurprisal.jsonl --out-path data/output/450_qwen3_30b_base_output.jsonl

nohup python -m surprisal_script --model-name deepseek-ai/deepseek-llm-7b-base --in-path data/input/450_llm_input_forsurprisal.jsonl --out-path data/output/450_deepseek_llm_7b_base_output.jsonl

nohup python -m surprisal_script --model-name deepseek-ai/deepseek-llm-67b-base --in-path data/input/450_llm_input_forsurprisal.jsonl --out-path data/output/450_deepseek_llm_67b_base_output.jsonl

"""

def run_inference(in_path, out_path, mol, batch_size=32, scale=DEFAULT_SCALE):
    with jsonl.open(in_path, 'r') as reader:
        with jsonl.open(out_path, 'a') as writer:
            batch = []
            for obj in reader:
                batch.append(obj)
                if len(batch) >= batch_size:
                    # Process and write batch
                    for item in get_surprisal_values_batch(batch, mol, scale):
                        writer.write(item)
                    batch = []

            # Process remaining items in batch
            if batch:
                for item in get_surprisal_values_batch(batch, mol, scale):
                    writer.write(item)


def make_sentence(obj, scale=DEFAULT_SCALE) -> Sentence:
    interventions = [Intervention(text=interv) for interv in get_interventions(scale)]
    context = obj['context']
    target = obj['target']
    # "often" interventions are bare adverbs and need the *_often.jsonl inputs (no trailing "it is ")
    if (scale == "often") == context.rstrip().endswith("it is"):
        raise ValueError(f"Context does not match --scale {scale} (use the *_often.jsonl input "
                         f"only with --scale often): {context!r}")
    return Sentence(
        context=context,
        target=target,
        intervened_target=target,
        interventions=interventions
    )


def add_measures(obj, sentence: Sentence, scale=DEFAULT_SCALE):
    intervention_dict = get_intervention_dict(scale)
    to_update = {}
    for intervention in sentence.interventions:
        txt = intervention.text
        surprisal = intervention.measures["surprisal"]
        toks = sentence.measure_tokens
        interv_key = intervention_dict[txt]
        to_update[f"{interv_key}_surprisal"] = surprisal
        to_update[f"{interv_key}_toks"] = toks
    to_update["intervention_scale"] = scale
    obj.update(to_update)
    return obj


def get_surprisal_values_batch(objs, mol, scale=DEFAULT_SCALE):
    # the HF backend scores these one sentence at a time, the vLLM backend all at once
    sentences = mol.get_measures_for_sentences([make_sentence(obj, scale) for obj in objs])
    return [add_measures(obj, sentence, scale) for obj, sentence in zip(objs, sentences)]


def get_surprisal_values(obj, mol, scale=DEFAULT_SCALE):
    return get_surprisal_values_batch([obj], mol, scale)[0]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Get surprisal values for input data")
    add_model_args(ap)
    ap.add_argument("--in-path", required=False, default=None)
    ap.add_argument("--out-path", required=True)
    ap.add_argument("--scale", choices=list(SCALES), default=DEFAULT_SCALE,
                    help=f"Intervention scale inserted between context and target (default: {DEFAULT_SCALE})")
    ap.add_argument("--batch-size", type=int, default=32,
                    help="Items per batch (default: 32). With --backend vllm use a much larger value "
                         "(e.g. 1000): each batch is one vLLM call")

    args = ap.parse_args()

    model = load_model(args)

    run_inference(in_path=args.in_path,
                  out_path=args.out_path,
                  mol=model,
                  batch_size=args.batch_size,
                  scale=args.scale)
    exit_cleanly(model)
