import argparse
import importlib.util
import os
from functools import lru_cache

import jsonlines as jsonl
import torch

from tools.load_model import add_model_args, load_model
from forced_choice.restricted import constrained_surprisals, renormalized_surprisals
from forced_choice.variables import (
    CHOICE_JOINER,
    DEFAULT_SCALE,
    PROMPT_JOINER,
    SCALES,
    SYSTEM_PROMPT,
    get_answer_choice_dict,
    get_answer_choices,
    get_options_instruction,
)

"""
Forced-choice version of surprisal_script.py.

Reads a jsonlines file where each line looks like

    {"unique_id": "t3125nex", "condition": "null_pos", "prompt": "... how likely is it that Emīls accepts?"}

appends the options instruction for --scale (default: likely) to the prompt, and measures the surprisal of each
answer choice (Very Unlikely, Unlikely, Likely, Very Likely) continuing that prompt:

    ... how likely is it that Emīls accepts? Choose one of the following options: Very Unlikely, Unlikely, Likely, Very Likely.
    Very Unlikely

For an instruction-tuned model pass --chat-template, which wraps that prompt in the
model's own chat template and scores the choice at the start of the assistant turn:

nohup python -m forced_choice.forced_choice_script --model-name /public/hf/models/meta-llama/Meta-Llama-3.1-8B-Instruct --in-path data/input/450_llm_prompt_input.jsonl --out-path data/forced_choice_output/Llama-3.1-8B-Instruct.jsonl --chat-template

Add --backend vllm (plus e.g. --tensor-parallel-size 4 --batch-size 500) to score with vLLM instead
of transformers. To compare the two on the same weights, run the HF backend with --no-load-in-8bit.

DeepSeek-V4 has no Jinja chat template; with --chat-template the script uses the model's own
encoding/encoding_dsv4.py in "chat" mode (thinking off, prompt ends with "<｜Assistant｜></think>").
vLLM's DeepSeek-V4 kernels need Hopper/Blackwell GPUs (not A100):

nohup python -m forced_choice.forced_choice_script --model-name /models/DeepSeek-V4-Flash-0731 --backend vllm --tensor-parallel-size 4 --batch-size 500 --chat-template --in-path data/input/for_llm_inference/normexp_input_likely.jsonl --out-path data/forced_choice_output/DeepSeek-V4-Flash-0731_likely.jsonl

Base models (no --chat-template):

nohup python -m forced_choice.forced_choice_script --model-name /public/hf/models/meta-llama/Meta-Llama-3.1-8B --model-nickname Llama3.1-8B --in-path data/input/450_llm_prompt_input.jsonl --out-path data/forced_choice_output/Llama-3.1-8B.jsonl > llama_8b_fc.out &!

nohup python -m forced_choice.forced_choice_script --model-name /public/hf/models/google/gemma-3-12b-pt --model-nickname gemma-3-12b --in-path data/input/450_llm_prompt_input.jsonl --out-path data/forced_choice_output/gemma-3-12b-pt.jsonl

nohup python -m forced_choice.forced_choice_script --model-name allenai/Olmo-3-1025-7B --in-path data/input/450_llm_prompt_input.jsonl --out-path data/forced_choice_output/Olmo-3-1025-7B.jsonl

nohup python -m forced_choice.forced_choice_script --model-name Qwen/Qwen3-8B-Base --in-path data/input/450_llm_prompt_input.jsonl --out-path data/forced_choice_output/Qwen3-8B-Base.jsonl

nohup python -m forced_choice.forced_choice_script --model-name deepseek-ai/deepseek-llm-7b-base --in-path data/input/450_llm_prompt_input.jsonl --out-path data/forced_choice_output/deepseek-llm-7b-base.jsonl

"""


# Some instruct models ship a Python encoder instead of a Jinja chat template. DeepSeek-V4 has
# encoding/encoding_dsv4.py with encode_messages(messages, thinking_mode="chat"|"thinking"):
# "chat" ends the prompt with "<｜Assistant｜></think>" (thinking block closed, answer directly),
# "thinking" with "<｜Assistant｜><think>". The encoded prompt already starts with the BOS token.
CUSTOM_ENCODERS = ["encoding/encoding_dsv4.py"]


@lru_cache(maxsize=None)
def load_custom_encoder(model_dir_or_id: str):
    """The model's own encode_messages module, or None if it has none of CUSTOM_ENCODERS.

    This runs Python code shipped with the model (like trust_remote_code), so only use it with
    checkpoints you trust.
    """
    for rel in CUSTOM_ENCODERS:
        if os.path.isdir(model_dir_or_id):
            path = os.path.join(model_dir_or_id, rel)
            if not os.path.isfile(path):
                continue
        else:
            from huggingface_hub import hf_hub_download
            try:
                path = hf_hub_download(model_dir_or_id, rel)
            except Exception:
                continue
        spec = importlib.util.spec_from_file_location(f"custom_encoder_{len(rel)}", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        print(f"Using the model's own prompt encoder: {path}")
        return module
    return None


def get_custom_encoder(tokenizer):
    """Only used when the tokenizer has no chat template."""
    if tokenizer is None or getattr(tokenizer, "chat_template", None) is not None:
        return None
    return load_custom_encoder(tokenizer.name_or_path)


def build_prompt(prompt: str, tokenizer=None, chat_template: bool = False,
                 system_prompt: str = SYSTEM_PROMPT, enable_thinking: bool = False,
                 scale: str = DEFAULT_SCALE) -> str:
    """The prompt from the input file with the answer options appended.

    For an instruction-tuned model, pass chat_template=True: the result is wrapped in
    the model's own chat template and ends with the generation prompt, so the answer
    choice is scored where the assistant turn starts.
    """
    text = f"{prompt.rstrip()}{PROMPT_JOINER}{get_options_instruction(scale)}"
    if not chat_template:
        return text
    messages = [{"role": "system", "content": system_prompt}] if system_prompt else []
    messages.append({"role": "user", "content": text})
    encoder = get_custom_encoder(tokenizer)
    if encoder is not None:
        return encoder.encode_messages(messages, thinking_mode="thinking" if enable_thinking else "chat")
    # enable_thinking is only read by templates that have a thinking mode (e.g. Qwen3);
    # other templates ignore it. False makes the model answer immediately rather than
    # opening a <think> block, which is what we want when scoring the answer directly.
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def n_choice_tokens(tokenizer, prompt: str, choice: str, add_special_tokens: bool = True) -> int:
    """How many tokens of `prompt + choice` belong to the choice.

    Computed as the tokens after the common prefix with the tokenized prompt, so a
    token that merges across the prompt/choice boundary is counted with the choice.
    """
    prompt_ids = tokenizer(prompt, add_special_tokens=add_special_tokens)["input_ids"]
    full_ids = tokenizer(f"{prompt}{choice}", add_special_tokens=add_special_tokens)["input_ids"]
    i = 0
    while i < len(prompt_ids) and i < len(full_ids) and prompt_ids[i] == full_ids[i]:
        i += 1
    return len(full_ids) - i


def get_choice_surprisals(mol, prompts: list[str], choices: list[str], max_length: int = 2048,
                          add_special_tokens: bool = True):
    """Per-token surprisal of each choice given its prompt.

    `prompts` and `choices` are parallel lists: one forward pass scores all of them.
    Returns a list of (token_strings, surprisals) pairs, one per pair of inputs.
    """
    tokenizer = mol.tokenizer
    n_toks = [n_choice_tokens(tokenizer, p, c, add_special_tokens=add_special_tokens)
              for p, c in zip(prompts, choices)]
    inputs = [f"{p}{c}" for p, c in zip(prompts, choices)]

    if mol.backend == "vllm":
        return get_choice_surprisals_vllm(mol, inputs, n_toks, max_length, add_special_tokens)

    batch = tokenizer(
        inputs,
        padding=True,
        truncation=True,
        padding_side="right",
        max_length=max_length,
        add_special_tokens=add_special_tokens,
        return_tensors="pt",
    ).to(mol.model.device)

    with torch.no_grad():
        outputs = mol.model(**batch)

    input_ids = batch.input_ids
    lengths = batch.attention_mask.sum(dim=-1)  # (B,) number of non-pad tokens

    # log_softmax in fp32 is a good idea even in bf16
    logprobs = torch.nn.functional.log_softmax(outputs.logits.float(), dim=-1)

    # Proper next-token alignment
    logprobs_aligned = logprobs[:, :-1, :]  # (B, T-1, V)
    target_ids = input_ids[:, 1:]  # (B, T-1)

    surprisals = -logprobs_aligned.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)  # (B, T-1)

    results = []
    for b in range(surprisals.size(0)):
        L = int(lengths[b].item())
        n = n_toks[b]
        if L >= max_length:
            print(f"WARNING: input {b} was truncated at {max_length} tokens, its surprisals are wrong")
        end = L - 1  # surprisals are defined for targets 1..L-1 -> length L-1
        start = max(0, end - n)
        toks = tokenizer.convert_ids_to_tokens(input_ids[b, start + 1:end + 1].tolist())
        results.append((toks, surprisals[b, start:end].tolist()))
    return results


def get_choice_surprisals_vllm(mol, inputs: list[str], n_toks: list[int], max_length: int,
                               add_special_tokens: bool):
    """vLLM version of the scoring in get_choice_surprisals: same tokenization and slicing,
    surprisals from vLLM's prompt_logprobs (see tools/vllm_model_of_language.py)."""
    tokenizer = mol.tokenizer
    id_lists = [tokenizer(x, truncation=True, max_length=max_length,
                          add_special_tokens=add_special_tokens)["input_ids"] for x in inputs]
    all_surprisals = mol.token_surprisals(id_lists)  # element k = surprisal of token k+1
    results = []
    for b, (ids, surprisals) in enumerate(zip(id_lists, all_surprisals)):
        L = len(ids)
        n = n_toks[b]
        if L >= max_length:
            print(f"WARNING: input {b} was truncated at {max_length} tokens, its surprisals are wrong")
        end = L - 1
        start = max(0, end - n)
        toks = tokenizer.convert_ids_to_tokens(ids[start + 1:end + 1])
        results.append((toks, surprisals[start:end]))
    return results


def choice_merges_with_prompt(tokenizer, prompt: str, choice: str, add_special_tokens: bool) -> bool:
    """True if tokenizing prompt + choice changes the prompt's own tokens (a token spans the
    boundary). Then the choices don't share an identical prompt, which constrained scoring assumes."""
    prompt_ids = tokenizer(prompt, add_special_tokens=add_special_tokens)["input_ids"]
    full_ids = tokenizer(f"{prompt}{choice}", add_special_tokens=add_special_tokens)["input_ids"]
    return full_ids[:len(prompt_ids)] != prompt_ids


def add_restricted_surprisals(to_update: dict, mergeable: bool, scale: str = DEFAULT_SCALE):
    """Add the two restricted versions (see forced_choice/restricted.py) for one item:
    {k}_renorm_surprisal: summed surprisal renormalized over the choices (same ranking as summed)
    {k}_constrained_surprisal: summed surprisal under token-level restriction to the choices,
    with {k}_constrained_token_surprisal per token. exp(-x) of either sums to 1 over the choices.
    """
    keys = list(get_answer_choice_dict(scale).values())
    summed = [to_update[f"{k}_sum_surprisal"] for k in keys]
    for k, v in zip(keys, renormalized_surprisals(summed)):
        to_update[f"{k}_renorm_surprisal"] = v
    if mergeable:
        for k in keys:
            to_update[f"{k}_constrained_surprisal"] = None
            to_update[f"{k}_constrained_token_surprisal"] = None
        return
    constrained = constrained_surprisals([to_update[f"{k}_toks"] for k in keys],
                                         [to_update[f"{k}_surprisal"] for k in keys])
    for k, per_token in zip(keys, constrained):
        to_update[f"{k}_constrained_surprisal"] = sum(per_token)
        to_update[f"{k}_constrained_token_surprisal"] = per_token


def check_prompt_scale(prompt: str, scale: str):
    """The question in the prompt must be on the same scale as the answer options."""
    if f"how {scale} is it" not in prompt:
        raise ValueError(f"--scale {scale}, but the prompt does not ask 'how {scale} is it': {prompt!r}")


def get_surprisal_values(objs, mol, max_length=2048, chat_template=False,
                         system_prompt=SYSTEM_PROMPT, enable_thinking=False, scale=DEFAULT_SCALE):
    """Score every answer choice for every object in `objs` in one forward pass."""
    ANSWER_CHOICES = get_answer_choices(scale)
    ANSWER_CHOICE_DICT = get_answer_choice_dict(scale)
    for obj in objs:
        check_prompt_scale(obj["prompt"], scale)
    full_prompts = [build_prompt(obj["prompt"], tokenizer=mol.tokenizer, chat_template=chat_template,
                                 system_prompt=system_prompt, enable_thinking=enable_thinking,
                                 scale=scale)
                    for obj in objs]

    # flat list of (prompt, choice) pairs: len(objs) * len(ANSWER_CHOICES) sequences
    batch_prompts = [p for p in full_prompts for _ in ANSWER_CHOICES]
    batch_choices = [f"{CHOICE_JOINER}{choice}" for _ in objs for choice in ANSWER_CHOICES]

    # a chat template already contains the model's BOS token, so don't let the
    # tokenizer add a second one
    results = get_choice_surprisals(mol, batch_prompts, batch_choices, max_length=max_length,
                                    add_special_tokens=not chat_template)

    n_choices = len(ANSWER_CHOICES)
    for i, obj in enumerate(objs):
        to_update = {"full_prompt": full_prompts[i], "answer_scale": scale}
        for j, choice in enumerate(ANSWER_CHOICES):
            toks, surprisal = results[i * n_choices + j]
            choice_key = ANSWER_CHOICE_DICT[choice]
            to_update[f"{choice_key}_choice"] = choice
            to_update[f"{choice_key}_surprisal"] = surprisal
            to_update[f"{choice_key}_toks"] = toks
            to_update[f"{choice_key}_sum_surprisal"] = sum(surprisal)
            to_update[f"{choice_key}_mean_surprisal"] = sum(surprisal) / len(surprisal) if surprisal else None
        mergeable = any(choice_merges_with_prompt(mol.tokenizer, full_prompts[i], f"{CHOICE_JOINER}{c}",
                                                  add_special_tokens=not chat_template)
                        for c in ANSWER_CHOICES)
        if mergeable:
            print(f"WARNING: an answer choice merges with the end of the prompt for "
                  f"{obj.get('unique_id')} {obj.get('condition')}; constrained surprisal set to None")
        add_restricted_surprisals(to_update, mergeable, scale)
        obj.update(to_update)
    return objs


def check_thinking_block(mol, chat_template: bool, system_prompt: str, enable_thinking: bool):
    """Warn if the generation prompt ends inside an unclosed thinking block.

    Not every template honours enable_thinking: Olmo-3-*-Think, for one, hardcodes a
    trailing '<think>' and ignores the flag, so the answer choices would be scored as
    the opening tokens of the model's reasoning rather than as its answer.
    """
    if not chat_template or enable_thinking:
        return
    probe = build_prompt("probe", tokenizer=mol.tokenizer, chat_template=True,
                         system_prompt=system_prompt, enable_thinking=False)
    if probe.count("<think>") > probe.count("</think>"):
        print(f"WARNING: {mol.nickname} opens a thinking block that enable_thinking=False did not "
              f"close, so its prompt ends with {probe[-40:]!r}. The answer choices will be scored "
              f"as the first tokens of the model's reasoning, not as its answer, and are not "
              f"comparable to the other models. Consider an -Instruct variant instead.")


def run_inference(in_path, out_path, mol, batch_size=8, max_length=2048, chat_template=False,
                  system_prompt=SYSTEM_PROMPT, enable_thinking=False, scale=DEFAULT_SCALE):
    if (chat_template and getattr(mol.tokenizer, "chat_template", None) is None
            and get_custom_encoder(mol.tokenizer) is None):
        raise ValueError(f"--chat-template was passed but {mol.name} has no chat template "
                         "or prompt encoder (is it a base model?)")
    check_thinking_block(mol, chat_template, system_prompt, enable_thinking)
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    kwargs = dict(max_length=max_length, chat_template=chat_template,
                  system_prompt=system_prompt, enable_thinking=enable_thinking, scale=scale)
    with jsonl.open(in_path, 'r') as reader:
        with jsonl.open(out_path, 'a') as writer:
            batch = []
            for obj in reader:
                batch.append(obj)
                if len(batch) >= batch_size:
                    # Process and write batch
                    for item in get_surprisal_values(batch, mol, **kwargs):
                        writer.write(item)
                    batch = []

            # Process remaining items in batch
            if batch:
                for item in get_surprisal_values(batch, mol, **kwargs):
                    writer.write(item)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Get answer-choice surprisal values for input data")
    add_model_args(ap)
    ap.add_argument("--in-path", required=False, default=None)
    ap.add_argument("--out-path", required=True)
    ap.add_argument("--batch-size", type=int, default=8,
                    help="Prompts per forward pass, each scored against every answer choice (default: 8). "
                         "With --backend vllm use a much larger value (e.g. 500): each batch is one vLLM call")
    ap.add_argument("--max-length", type=int, default=2048, help="Truncation length in tokens (default: 2048)")
    ap.add_argument("--chat-template", action="store_true",
                    help="Wrap the prompt in the model's chat template. Use for instruction-tuned models")
    ap.add_argument("--system-prompt", required=False, default=SYSTEM_PROMPT,
                    help=f"System message, held constant across models. Only used with "
                         f"--chat-template. Pass \"\" to leave it out and let each chat template "
                         f"supply its own default (default: {SYSTEM_PROMPT!r})")
    ap.add_argument("--scale", choices=list(SCALES), default=DEFAULT_SCALE,
                    help=f"Answer options to offer and score; must match the question in the "
                         f"prompts, e.g. 'how often is it that' -> often (default: {DEFAULT_SCALE})")
    ap.add_argument("--enable-thinking", action="store_true",
                    help="For thinking models (e.g. Qwen3), leave the thinking block open instead of "
                         "closing it before the answer. Only used with --chat-template")

    args = ap.parse_args()

    model = load_model(args)

    run_inference(in_path=args.in_path,
                  out_path=args.out_path,
                  mol=model,
                  batch_size=args.batch_size,
                  max_length=args.max_length,
                  chat_template=args.chat_template,
                  system_prompt=args.system_prompt,
                  enable_thinking=args.enable_thinking,
                  scale=args.scale)
