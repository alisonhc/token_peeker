import argparse
import jsonlines as jsonl

from tools.model_of_language import *
from tools.variables import INTERVENTIONS, INTERVENTION_DICT

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
        surprisal, ent, kl_div = intervention.measures["surprisal"], intervention.measures["entropy"], intervention.measures["kl_div"]
        toks = sentence.measure_tokens
        interv_key = INTERVENTION_DICT[txt]
        to_update[f"{interv_key}_surprisal"] = surprisal
        to_update[f"{interv_key}_ent"] = ent
        to_update[f"{interv_key}_kl_div"] = kl_div
        to_update[f"{interv_key}_toks"] = toks
    obj.update(to_update)
    return obj


#mol = ModelOfLanguage(path='/public/hf/models/mistralai/Mistral-7B-v0.1', nickname="Mistral7B")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Get surprisal values for input data")

    # eg '/public/hf/models/mistralai/Mistral-7B-v0.1'
    ap.add_argument("--model-name", required=True)

    # e.g. "Mistral7B"
    ap.add_argument("--model-nickname", required=True)
    ap.add_argument("--in-path", required=True)
    ap.add_argument("--out-path", required=True)
    ap.add_argument("--batch-size", type=int, default=32, help="Batch size for processing (default: 32)")

    args = ap.parse_args()

    model = ModelOfLanguage(path=args.model_name, nickname=args.model_nickname)

    run_inference(in_path=args.in_path,
                  out_path=args.out_path,
                  mol=model,
                  batch_size=args.batch_size)