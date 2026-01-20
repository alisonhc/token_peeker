from typing import Optional, Any
import os
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import nltk
import spacy
import logging

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
    def __init__(self, nickname:str = None, name: str = None, path: str = None):
        self.name = self.set_name(nickname, name)
        # self.name = name
        self.nickname = self.set_nickname(nickname, name)
        self.path = path
        self.model_key = self.path if self.path else self.name
        print(f"Model nickname set to: {self.nickname}")
        print(f"Model name set to: {self.name}")
        print(f"Local model path: {self.path}")
        print(f"Model key set to {self.model_key}")

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
        elif nickname in ["Llama3.1", "Llama3.1-8B"]:
            return "meta-llama/Llama-3.1-8B"
        elif nickname in ["Llama3.1-Instruct"]:
            return "meta-llama/Llama-3.1-8B-Instruct"
        elif nickname in ["Mixtral8x22B", "8x22B"]:
            return "mistralai/Mixtral-8x22B-v0.1"
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
        pass
    
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