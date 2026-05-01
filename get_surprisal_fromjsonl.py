import argparse
import jsonlines as jsonl

from tools.model_of_language import *

mol = ModelOfLanguage(path='/public/hf/models/mistralai/Mistral-7B-v0.1', nickname="Mistral7B")

def run_inference(in_path, out_path, mol):
    with jsonl.open(in_path, 'r') as reader:
        with jsonl.open(out_path, 'a') as writer:
            for obj in reader: # TODO change to batch reading
                surprisal, entropy, kl_div = get_surprisal_values(obj)

def get_surprisal_values(obj):
    context = obj['context']
    target = obj['target']
    



if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Get surprisal values for a model on a specific dataset")
    ap.add_argument("--model", required=True)  # model name, '/public/hf/models/mistralai/Mistral-7B-v0.1'
    ap.add_argument("--model-nickname", required=True, help="nickname for model") # "Mistral7B""
    ap.add_argument("--input-path", required=True, help="jsonl path for input data")
    ap.add_argument("--output-path", required=True, help="jsonl output to write output")
    
