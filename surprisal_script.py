import argparse
import jsonlines as jsonl

from tools.model_of_language import *
from tools.variables import INTERVENTIONS, INTERVENTION_DICT

"""

nohup python -m surprisal_script --model-name /public/hf/models/meta-llama/Meta-Llama-3.1-8B --model-nickname Llama3.1-8B --in-path data/input/450_llm_input_forsurprisal.jsonl --out-path data/output/450_llama3.1_8b_output.jsonl > llama_8b_450.out &!

nohup python -m surprisal_script --model-name /public/hf/models/google/gemma-3-12b-pt --model-nickname gemma-3-12b --in-path data/input/450_llm_input_forsurprisal.jsonl --out-path data/output/450_gemma3_12b_output.jsonl

python -m surprisal_script --model-name /public/hf/models/google/gemma-3-27b-pt --model-nickname gemma-3-27b --in-path data/input/450_llm_input_forsurprisal.jsonl --out-path data/output/450_gemma3_27b_output.jsonl

python -m surprisal_script --model-name allenai/Olmo-3-1025-7B --in-path data/input/450_llm_input_forsurprisal.jsonl --out-path data/output/450_olmo3_7b_output.jsonl

python -m surprisal_script --model-name allenai/Olmo-3-1125-32B --in-path data/input/450_llm_input_forsurprisal.jsonl --out-path data/output/450_olmo3_32b_output.jsonl

python -m surprisal_script --model-name /public/hf/models/meta-llama/Meta-Llama-3.1-70B --in-path data/input/450_llm_input_forsurprisal.jsonl --out-path data/output/450_llama3.1_70b_output.jsonl

nohup python -m surprisal_script --model-name Qwen/Qwen3-8B-Base --in-path data/input/450_llm_input_forsurprisal.jsonl --out-path data/output/450_qwen3_8b_base_output.jsonl

python -m surprisal_script --model-name Qwen/Qwen3-30B-A3B-Base --in-path data/input/450_llm_input_forsurprisal.jsonl --out-path data/output/450_qwen3_30b_base_output.jsonl

python -m surprisal_script --model-name deepseek-ai/deepseek-llm-7b-base --in-path data/input/450_llm_input_forsurprisal.jsonl --out-path data/output/450_deepseek_llm_7b_base_output.jsonl

python -m surprisal_script --model-name deepseek-ai/deepseek-llm-67b-base --in-path data/input/450_llm_input_forsurprisal.jsonl --out-path data/output/450_deepseek_llm_67b_base_output.jsonl

"""

def run_inference(in_path, out_path, mol, batch_size=32):
    with jsonl.open(in_path, 'r') as reader:
        with jsonl.open(out_path, 'a') as writer:
            batch = []
            for obj in reader:
                batch.append(obj)
                if len(batch) >= batch_size:
                    # Process and write batch
                    for item in batch:
                        writer.write(get_surprisal_values(item, mol))
                    batch = []
            
            # Process remaining items in batch
            if batch:
                for item in batch:
                    writer.write(get_surprisal_values(item, mol))

def get_surprisal_values(obj, mol):
    interventions = [Intervention(text=interv) for interv in INTERVENTIONS]
    context = obj['context']
    target = obj['target']
    sentence = Sentence(
        context=context, 
        target=target, 
        intervened_target=target,
        interventions=interventions
    )
    measures = mol.get_measures_for_sentence(sentence)
    to_update = {}
    for intervention in measures.interventions:
        txt = intervention.text
        # surprisal, ent, kl_div = intervention.measures["surprisal"], intervention.measures["entropy"], intervention.measures["kl_div"]
        surprisal = intervention.measures["surprisal"]
        toks = sentence.measure_tokens
        interv_key = INTERVENTION_DICT[txt]
        to_update[f"{interv_key}_surprisal"] = surprisal
        # to_update[f"{interv_key}_ent"] = ent
        # to_update[f"{interv_key}_kl_div"] = kl_div
        to_update[f"{interv_key}_toks"] = toks
    obj.update(to_update)
    return obj


#mol = ModelOfLanguage(path='/public/hf/models/mistralai/Mistral-7B-v0.1', nickname="Mistral7B")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Get surprisal values for input data")

    # eg '/public/hf/models/mistralai/Mistral-7B-v0.1'
    ap.add_argument("--model-name", required=False, default=None)

    # e.g. "Mistral7B"
    ap.add_argument("--model-nickname", required=False, default=None)
    ap.add_argument("--in-path", required=False, default=None)
    ap.add_argument("--out-path", required=True)
    ap.add_argument("--batch-size", type=int, default=32, help="Batch size for processing (default: 32)")

    args = ap.parse_args()

    path_to_use = args.model_name if args.model_name and 'public' in args.model_name else None
    model = ModelOfLanguage(path=path_to_use,
                            nickname=args.model_nickname,
                            name=args.model_name,
                            use_vllm=False)

    run_inference(in_path=args.in_path,
                  out_path=args.out_path,
                  mol=model,
                  batch_size=args.batch_size)

