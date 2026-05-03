from typing import Optional, Any
import os
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import nltk
import spacy
import logging
import math

# Optional vLLM support (scaffolded). If vllm is installed, `VLLM_AVAILABLE` will be True.
try:
    from vllm import LLM, SamplingParams  # type: ignore
    VLLM_AVAILABLE = True
except Exception:
    LLM = None  # type: ignore
    SamplingParams = None  # type: ignore
    VLLM_AVAILABLE = False
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logging.info("Downloading spaCy model...")
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")



class Intervention(BaseModel):
    text: str
    measures: dict[str, list[float]] = {}
    
    
class Sentence(BaseModel):
    id: Optional[str] = None
    context: str
    original_quantifier: Optional[str] = None
    # this is always the generic sentence, regardless of the original quantifier, it is a bit awkward but it is what it is!
    target: str
    intervened_target: Optional[str] = None
    target_measures: Optional[dict[str, list[float]]] = None
    # context_cached_hidden_states: Optional[Any] = None
    
    # these are the common tokens between the target and the intervened target
    measure_tokens: Optional[list[str]] = None 
    # what is the index in the interventions of the original one, if 0 then generic
    original_intervention_index: Optional[int] = None
    # what is the first token after the root, in order to do p-acceptability-style calculations
    first_token_after_root_from_right: Optional[int] = None
    interventions: Optional[list[Intervention]] = None
    
    
    # if intervened_target is None, make it the target but with the first letter lowercased
    def __init__(self, **data):
        super().__init__(**data)
        if self.intervened_target is None:
            self.intervened_target = self.target[0].lower() + self.target[1:]
        if self.original_intervention_index is None:
            self.original_intervention_index = 0

    def _format_list(self, values: list[float] | None, max_items: int = 12) -> str:
        if not values:
            return "[]"
        shown = values[:max_items]
        formatted = ", ".join(f"{v:.4f}" for v in shown)
        if len(values) > max_items:
            return f"[{formatted}, ...] (n={len(values)})"
        return f"[{formatted}]"

    def pretty(self, show_values: bool = False, max_list_items: int = 12) -> str:
        lines: list[str] = []
        lines.append("Sentence")
        lines.append(f"- context: {self.context}")
        lines.append(f"- target: {self.target}")
        if self.intervened_target is not None:
            lines.append(f"- intervened_target: {self.intervened_target}")
        if self.original_quantifier is not None:
            lines.append(f"- original_quantifier: {self.original_quantifier}")
        if self.measure_tokens is not None:
            lines.append(f"- measure_tokens: {self.measure_tokens}")
        if self.first_token_after_root_from_right is not None:
            lines.append(
                f"- first_token_after_root_from_right: {self.first_token_after_root_from_right}"
            )
        if self.target_measures is not None:
            tm = self.target_measures
            lines.append("- target_measures:")

            lines.append(
                f"  - entropy: {self._format_list(tm.get('entropy'), max_list_items)}"
            )
            lines.append(
                f"  - surprisal: {self._format_list(tm.get('surprisal'), max_list_items)}"
            )
            lines.append(
                f"  - kl_div: {self._format_list(tm.get('kl_div'), max_list_items)}"
            )
        if self.interventions is not None and len(self.interventions) > 0:
            lines.append("- interventions:")
            for idx, intervention in enumerate(self.interventions):
                header = f"  [{idx}] {intervention.text}"
                if intervention.measures:
            
                    lines.append(header)
                    lines.append(
                        f"    - entropy: {self._format_list(intervention.measures.get('entropy'), max_list_items)}"
                    )
                    lines.append(
                        f"    - surprisal: {self._format_list(intervention.measures.get('surprisal'), max_list_items)}"
                    )
                    lines.append(
                        f"    - kl_div: {self._format_list(intervention.measures.get('kl_div'), max_list_items)}"
                    )
        return "\n".join(lines)

    def __str__(self) -> str:  # Allows print(sentence)
        return self.pretty(show_values=False)


def print_sentence(sentence: Sentence, show_values: bool = False, max_list_items: int = 12) -> None:
    print(sentence.pretty(show_values=show_values, max_list_items=max_list_items))


class ModelOfLanguage:
    def __init__(self, nickname: str = None, name: str = None, path: str = None, use_vllm: bool = False):
        self.name = self.set_name(nickname, name)
        # self.name = name
        self.nickname = self.set_nickname(nickname, name)
        self.path = path
        self.model_key = self.path if self.path else self.name
        print(f"Model nickname set to: {self.nickname}")
        print(f"Model name set to: {self.name}")
        print(f"Local model path: {self.path}")
        print(f"Model key set to {self.model_key}")

        # vLLM usage flag (optional)
        self.use_vllm = use_vllm and VLLM_AVAILABLE
        if use_vllm and not VLLM_AVAILABLE:
            logging.warning("vLLM requested but not available; falling back to transformers")

        # model objects: when using transformers `self.model` holds HF model;
        # when using vllm, `self.vllm` will hold the vllm.LLM instance and
        # `self.model` will be the HF model only as a fallback for parts
        # of the code that still use transformers APIs.
        self.vllm = None
        self.model = self.load_model()
        self.tokenizer = self.load_tokenizer()
        
    def set_name(self, nickname:str, name: str):
        if nickname is None and name is None:
            raise ValueError("Either nickname or name must be provided")
        if nickname is None:
            return name
        elif nickname == "gpt2":
            return "openai-community/gpt2"
        elif nickname == "Mistral7B":
            return "mistralai/Mistral-7B-v0.3"
        elif nickname == "Mistral7B-Instruct":
            return "mistralai/Mistral-7B-Instruct-v0.3"
        elif nickname in ["Qwen3-8B-Base", "Qwen3"]:
            return "Qwen/Qwen3-8B-Base"
        elif nickname == "Qwen3-Instruct":
            return "Qwen/Qwen3-8B"
        elif nickname == "Qwen3.5-9B-Base":
            return "Qwen/Qwen3.5-9B-Base"
        elif nickname in ["Llama3.1", "Llama3.1-8B"]:
            return "meta-llama/Llama-3.1-8B"
        elif nickname in ["Llama3.1-Instruct"]:
            return "meta-llama/Llama-3.1-8B-Instruct"
        elif nickname in ["Mixtral8x22B", "8x22B"]:
            return "mistralai/Mixtral-8x22B-v0.1"
        elif nickname == "gemma-3-12b":
            return "google/gemma-3-12b-pt"
        elif nickname == "gemma-3-27b":
            return "google/gemma-3-27b-pt"
        else:
            raise ValueError(f"Model {nickname} not found")

    def set_nickname(self, nickname:str, name: str):
        if nickname is None and name is None:
            raise ValueError("Either nickname or name must be provided")
        if nickname is None:
            return name.split("/")[-1]
        else:
            return nickname
        
    def load_tokenizer(self):
        pass
    
    def load_model(self):
        # try:
        if self.name == "openai-community/gpt2":
            model = AutoModelForCausalLM.from_pretrained(
                "openai-community/gpt2",
                device_map="auto",
            )  
        elif self.name == "mistralai/Mixtral-8x22B-v0.1":
            model = AutoModelForCausalLM.from_pretrained(
                "mistralai/Mixtral-8x22B-v0.1", 
                load_in_4bit=True,
                device_map="auto"
            )
        else:
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                self.model_key,
                quantization_config=quantization_config,
                device_map="auto",
            )
        # except ValueError:
        #     raise ValueError(f"Model {self.name} not found")
        return model
    
    def load_tokenizer(self):
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_key)
        except ValueError:
            raise ValueError(f"Model {self.model_key} not found")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    
    def get_activations(self, sentence: str):
        # Activation extraction for vLLM is different from HF transformers.
        # This method is left as a compatibility stub; use
        # `_get_hidden_states_for_context` or implement a vLLM-specific
        # extraction if needed.
        raise NotImplementedError("get_activations is not implemented for vLLM/stub")
    
    def get_hidden_states_for_context(self, context: str):
        with torch.no_grad():
            tokens = self.tokenizer(context, return_tensors="pt")
            outputs = self.model(**tokens, use_cache=True)
            hidden_states = outputs.past_key_values
            return hidden_states
    
    def get_common_tokens(self, text_a, text_b):
        """ Get the longest common subsequence of tokens between two texts, that includes the last token."""
        tokens_a = self.tokenizer.tokenize(text_a)
        tokens_b = self.tokenizer.tokenize(text_b)
        common_tokens = []
        # start from the end of both token lists and compare tokens
        for i in range(min(len(tokens_a), len(tokens_b))):
            if tokens_a[-(i+1)] == tokens_b[-(i+1)]:
                common_tokens.append(tokens_a[-(i+1)])
            else:
                break
        common_tokens.reverse()  # reverse to get the correct order
        return common_tokens
    
    def join_context_and_target(self, context: str, target: str):
        """Join context and target with a space, ensuring no leading or trailing spaces."""
        return f"{context} {target}".strip()
    
    def kl_divergences(self, probs_p, logprobs_p: torch.Tensor, logprobs_q: torch.Tensor):
        return torch.sum(probs_p * (logprobs_p - logprobs_q), dim=-1)
    
    def get_measures_at_target_tokens(self, targets: list[str], contexts: list[str], n_relevant_tokens_from_last: int, original_intervention_index: int=0):
        # If using vLLM, delegate to the vLLM-specific implementation.
        if self.use_vllm:
            return self._get_measures_with_vllm(targets, contexts, n_relevant_tokens_from_last, original_intervention_index)

        inputs = [self.join_context_and_target(c,t) for c, t in zip(contexts, targets)]
        token_ids = self.tokenizer(
            inputs, 
            padding=True,
            truncation=True,
            padding_side="right",
            max_length=1024,
            return_tensors="pt"
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model(**token_ids)
            
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        logprobs = torch.nn.functional.log_softmax(outputs.logits, dim=-1)
        entropies = -torch.sum(probs * logprobs, dim=-1)
        surprisals = -logprobs.gather(-1, token_ids.input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
        last_relevant_token = torch.sum(token_ids.attention_mask, dim=-1) - len(token_ids.input_ids[0])
        surprisals_list = [s[-n_relevant_tokens_from_last+i:i].tolist() if i != 0 else s[-n_relevant_tokens_from_last:].tolist() for i, s in zip(last_relevant_token, surprisals)]
    
        
        
        # entropies have to be shifted by 1 to the left, becasue in the surprisals, from the distribution on the first token we derive the surprisal of the second token, but this is done by passing token_ids.input_ids[:, 1:]. in the case of the entropy, we need to do this shift afterwards
        entropies_list = [
            e[-n_relevant_tokens_from_last+i-1:i-1].tolist() if i != 0 else e[-n_relevant_tokens_from_last-1:-1].tolist() for i, e in zip(last_relevant_token, entropies)
        ]
        
        # kl divergences wrt to the first sentence
        kl_divs_list = []
        # print(f"last_relevant_token: {last_relevant_token}")
        for j, i in enumerate(last_relevant_token):
            # KL(P,Q), P the true distribution, Q the wrong distribution
            kl_divs = self.kl_divergences(
                # original quantifier intervention probs
                probs[original_intervention_index][-n_relevant_tokens_from_last+last_relevant_token[original_intervention_index]-1:last_relevant_token[original_intervention_index]-1,:],
                logprobs[original_intervention_index][-n_relevant_tokens_from_last+last_relevant_token[original_intervention_index]-1:last_relevant_token[original_intervention_index]-1,:],
                
                # wrong quantifier 
                logprobs[j][-n_relevant_tokens_from_last+i-1:i-1,:] if i != 0 else logprobs[j][-n_relevant_tokens_from_last-1:-1,:],
            )
            kl_divs_list.append(kl_divs.tolist())
        
        return {
            "entropy": entropies_list,
            "surprisal": surprisals_list,
            "kl_div": kl_divs_list,
        }

    def _get_measures_with_vllm(self, targets: list[str], contexts: list[str], n_relevant_tokens_from_last: int, original_intervention_index: int = 0):
        """Scaffold for computing measures using vLLM.

        vLLM exposes a different runtime API than Hugging Face transformers.
        The implementation here is intentionally a scaffold: vLLM can return
        per-token logits via the streaming/generation API and model outputs,
        but the exact extraction depends on the vllm version and usage
        pattern (e.g., `LLM.generate` and inspecting `response.output`).

        If you want full vLLM support, implement this to:
        - tokenize `contexts + targets` with the HF tokenizer or vLLM tokenizer
        - run vLLM generation with a request to return logits/logprobs
        - compute `probs`, `logprobs`, `entropies`, `surprisals`, and KL
        - return the same dict shape as the transformers path

        For now this raises `NotImplementedError` to keep behavior explicit.
        """
        def _to_finite_float(value, default=None):
            try:
                f = float(value)
            except Exception:
                return default
            if not math.isfinite(f):
                return default
            return f

        def _sanitize_logprob_dict(entry):
            if not isinstance(entry, dict):
                return None
            clean = {}
            for k, v in entry.items():
                fv = _to_finite_float(v, default=None)
                if fv is None:
                    continue
                clean[k] = fv
            if len(clean) == 0:
                return None
            max_log = max(clean.values())
            exps = {}
            for k, v in clean.items():
                # Clamp exponent input to prevent overflow/underflow issues.
                exps[k] = math.exp(max(-80.0, min(80.0, v - max_log)))
            norm = sum(exps.values())
            if norm <= 0.0 or not math.isfinite(norm):
                return None
            return {k: exps[k] / norm for k in exps}

        def _lookup_token_logprob(entry, token_id):
            if not isinstance(entry, dict):
                return None
            if token_id in entry:
                return _to_finite_float(entry[token_id], default=None)
            # Common mismatch: token keys may be strings.
            token_id_str = str(token_id)
            if token_id_str in entry:
                return _to_finite_float(entry[token_id_str], default=None)
            return None

        # Lazy-initialize vLLM engine if not already created
        if not VLLM_AVAILABLE:
            raise RuntimeError("vLLM is not available in this environment")
        if getattr(self, "vllm", None) is None:
            # Default init; users can adjust if they want different engine args
            self.vllm = LLM(model=self.model_key)

        prompts = [self.join_context_and_target(c, t) for c, t in zip(contexts, targets)]

        # Request prompt-only logprobs by setting max_tokens=0 and temperature=0 (deterministic)
        sampling = SamplingParams(temperature=0.0)
        outputs = self.vllm.generate(prompts, sampling_params=sampling, max_tokens=0)

        # We'll build per-input lists of entropies and surprisals, and keep per-position distributions
        all_entropies = []
        all_surprisals = []
        all_distributions = []  # list of list-of-dicts mapping token_id->prob for each position

        for out in outputs:
            # Extract prompt token ids and prompt logprobs if available
            token_ids = None
            prompt_logprobs = None
            try:
                token_ids = getattr(out, "prompt_token_ids", None)
                prompt_logprobs = getattr(out, "prompt_logprobs", None)
            except Exception:
                token_ids = None
                prompt_logprobs = None

            # Fallback: try to inspect first completion output
            if token_ids is None:
                try:
                    comp = out.outputs[0]
                    token_ids = getattr(comp, "token_ids", None)
                except Exception:
                    token_ids = None

            entropies_pos = []
            surprisals_pos = []
            dists_pos = []

            if token_ids is None:
                # Can't extract token-level measures; return empty lists
                all_entropies.append([])
                all_surprisals.append([])
                all_distributions.append([])
                continue

            # Iterate positions and try to build a full distribution from prompt_logprobs
            for i, tid in enumerate(token_ids):
                logp_obs = None
                dist = None

                if prompt_logprobs is not None:
                    try:
                        entry = prompt_logprobs[i]
                    except Exception:
                        entry = None

                    # If entry is a dict mapping token_id->logprob
                    if isinstance(entry, dict):
                        dist = _sanitize_logprob_dict(entry)
                        logp_obs = _lookup_token_logprob(entry, tid)
                    elif isinstance(entry, float) or isinstance(entry, int):
                        # If entry is a scalar, assume it's the logprob of the observed token
                        logp_obs = _to_finite_float(entry, default=None)
                        if logp_obs is not None:
                            dist = {int(tid): 1.0}
                    else:
                        # Unknown entry format; attempt best-effort extraction
                        try:
                            # Some vLLM versions may expose token_logprobs as a list of (ids, logps)
                            token_logprobs = getattr(prompt_logprobs, "token_logprobs", None)
                            if token_logprobs is not None:
                                entry2 = token_logprobs[i]
                                if isinstance(entry2, dict):
                                    dist = _sanitize_logprob_dict(entry2)
                                    logp_obs = _lookup_token_logprob(entry2, tid)
                        except Exception:
                            dist = None

                # If we couldn't build a distribution, fall back to using any completion-level logprobs
                if dist is None:
                    # try completion outputs
                    try:
                        comp = out.outputs[0]
                        comp_logprobs = getattr(comp, "logprobs", None) or getattr(comp, "token_logprobs", None)
                        if comp_logprobs is not None and i < len(comp_logprobs):
                            val = comp_logprobs[i]
                            if isinstance(val, dict):
                                dist = _sanitize_logprob_dict(val)
                                logp_obs = _lookup_token_logprob(val, tid)
                            elif isinstance(val, float) or isinstance(val, int):
                                logp_obs = _to_finite_float(val, default=None)
                                if logp_obs is not None:
                                    dist = {int(tid): 1.0}
                    except Exception:
                        pass

                # Final fallbacks
                if logp_obs is None:
                    # If still missing, set to a very small probability (log prob large negative)
                    logp_obs = math.log(1e-12)
                if dist is None:
                    # represent degenerate distribution concentrated on the observed token
                    dist = {int(tid): 1.0}

                # Compute surprisal for this position (we'll shift later to align)
                surprisal = -float(logp_obs)

                # Compute entropy for this position from dist
                entropy = 0.0
                for p in dist.values():
                    if p > 0.0 and math.isfinite(p):
                        entropy -= p * math.log(p)

                if not math.isfinite(entropy):
                    entropy = 0.0
                if not math.isfinite(surprisal):
                    surprisal = -math.log(1e-12)

                entropies_pos.append(entropy)
                surprisals_pos.append(surprisal)
                dists_pos.append(dist)

            # Align entropies and surprisals like the transformers path: entropy at position i predicts token i+1
            # So we drop the last entropy to align with surprisals (which are computed for tokens 1..L-1)
            if len(entropies_pos) > 0:
                entropies_shifted = entropies_pos[:-1]
            else:
                entropies_shifted = []

            # surprisals: skip the first token because it has no preceding distribution
            surprisals_aligned = surprisals_pos[1:] if len(surprisals_pos) > 1 else []

            all_entropies.append(entropies_shifted)
            all_surprisals.append(surprisals_aligned)
            all_distributions.append(dists_pos)

        # Now compute kl divergences per input relative to the original_intervention_index using the distributions we built
        kl_divs_list = []
        eps = 1e-12
        ref_dists = all_distributions[original_intervention_index] if len(all_distributions) > original_intervention_index else []

        for j, dists in enumerate(all_distributions):
            kl_per_pos = []
            # Compare positions up to the min length of ref & current (use shifted alignment: we used dists_pos as full prompt-level dists)
            max_pos = min(len(ref_dists), len(dists))
            for pos in range(max_pos):
                p_dist = ref_dists[pos]
                q_dist = dists[pos]
                kl = 0.0
                for tok, p in p_dist.items():
                    if not math.isfinite(p) or p <= 0.0:
                        continue
                    q = q_dist.get(tok, eps)
                    q = q if (math.isfinite(q) and q > 0.0) else eps
                    if p > 0.0:
                        kl += p * (math.log(p + eps) - math.log(q + eps))
                if not math.isfinite(kl):
                    kl = 0.0
                kl_per_pos.append(kl)
            kl_divs_list.append(kl_per_pos)

        # Trim/slice to keep only the last `n_relevant_tokens_from_last` positions, matching the transformers path behavior
        def tail_slice(list_of_lists, n):
            out = []
            for lst in list_of_lists:
                if not lst:
                    out.append([])
                    continue
                out.append(lst[-n:])
            return out

        entropies_list = tail_slice(all_entropies, n_relevant_tokens_from_last)
        surprisals_list = tail_slice(all_surprisals, n_relevant_tokens_from_last)
        kl_divs_list = tail_slice(kl_divs_list, n_relevant_tokens_from_last)

        return {
            "entropy": entropies_list,
            "surprisal": surprisals_list,
            "kl_div": kl_divs_list,
        }
        
    def pointwise_list_subtraction(self, list_a, list_b):
        return [a - b for a, b in zip(list_a, list_b)]
    
    def mean(self, list_a):
        return sum(list_a)/len(list_a)
        
    def get_measures_for_sentence(self, sentence:Sentence):
        targets = [sentence.target] + [self.join_context_and_target(i.text, sentence.intervened_target ) for i in sentence.interventions] if sentence.interventions else [sentence.target]
        # print(f"Targets: {targets}")
        contexts = [sentence.context] * len(targets)
        if sentence.measure_tokens is None:
            sentence.measure_tokens = self.get_common_tokens(sentence.target, sentence.intervened_target)
        sentence.first_token_after_root_from_right = self.get_root_token_position(sentence.target)
        measures = self.get_measures_at_target_tokens(
            targets=targets,
            contexts=contexts,
            n_relevant_tokens_from_last=len(sentence.measure_tokens),
            original_intervention_index=sentence.original_intervention_index
        )
        sentence.target_measures = {
            "entropy": measures["entropy"][0],
            "surprisal": measures["surprisal"][0],
            "kl_div": measures["kl_div"][0],
            "typicality": self.mean(measures["surprisal"][0])-self.mean(measures["entropy"][0])
        }
        for i, intervention in enumerate(sentence.interventions):
            intervention.measures = {
                "entropy": measures["entropy"][i+1],
                "surprisal": measures["surprisal"][i+1],
                "kl_div": measures["kl_div"][i+1],
                "typicality": self.mean(measures["surprisal"][i+1])-self.mean(measures["entropy"][i+1])
            }
        return sentence
    
    def get_root_token_position(self,text):
        doc = nlp(text)
        root_word = (
            [token for token in doc if token.dep_ == "ROOT"][0].text
            if len([token for token in doc if token.dep_ == "ROOT"]) > 0
            else None
        )


        tokenizer_tokens = self.tokenizer.convert_ids_to_tokens(
            [t for t in self.tokenizer(text)["input_ids"] if t not in [self.tokenizer.pad_token_id, self.tokenizer.bos_token_id]]
        )

        min_distance = float("inf")
        min_distance_i = -1
        root_token = ""
        root_token_position = -1

        for i, token in enumerate(tokenizer_tokens):
            distance = nltk.edit_distance(token.replace("▁", "").replace("Ġ",""), root_word)
            if distance == 0:
                min_distance_i = i
                min_distance = distance
                root_token = token
                root_token_position = i
                break
            elif distance < min_distance:
                min_distance_i = i
                min_distance = distance
                root_token = token
                root_token_position = i

        # get last token of the root word
        while "▁" in root_token or "Ġ" in root_token:
            if len(tokenizer_tokens) > root_token_position+1 and "▁" not in tokenizer_tokens[root_token_position+1] and "Ġ" not in tokenizer_tokens[root_token_position+1]:
                root_token_position += 1
                root_token = tokenizer_tokens[root_token_position]
            else:
                break
        first_token_after_root_from_right = len(tokenizer_tokens) - root_token_position-1
        return first_token_after_root_from_right #root_token_position, tokenizer_tokens[-first_token_after_root_from_right:]