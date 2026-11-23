"""Surprisal of each answer choice when the model may only output one of the choices.

forced_choice_script.py scores each choice token by token over the full vocabulary:
    surprisal(choice) = sum_i -log p(t_i | prompt, t_<i)
This module turns those per-token values into two restricted versions, without extra model calls.

1. "renormalized": sequence-level renormalization over the choices.
       p_r(c) = p(c) / sum_c' p(c')      surprisal_r(c) = surprisal(c) + logsumexp_c'(-surprisal(c'))
   The four probabilities sum to 1, so they are comparable across items. But the term added is the
   same for every choice of an item, so within an item the ranking and the differences between
   choices are exactly those of the summed surprisal: the length handicap of longer labels stays.

2. "constrained": token-level restriction, i.e. what constrained decoding to the four choices does.
   The choices form a prefix tree of tokens. At every token, the model's probability is
   renormalized over only the tokens that continue some choice from that point:
       p_c(t_i | prefix) = p(t_i | prefix) / sum_{t allowed after prefix} p(t | prefix)
   A token that is the only allowed continuation (e.g. "likely" after "Very Un") gets
   probability 1, surprisal 0, so tokens that merely complete a label cost nothing. The choice's
   probabilities again sum to 1 over the four choices, and longer labels are no longer penalized
   just for having more tokens; only the branching decisions count, e.g. for the likely scale:
       "Very" vs "Un" vs "Lik" at the first token, then " Un" vs " Lik" after "Very".

Every probability needed for (2) is already in the per-token surprisals: all choices share the
prompt, so the logprob of an allowed token after a shared prefix is the same no matter which
choice's sequence it was read from (taken from the first choice through that token).
"""

import math


def logsumexp(xs: list[float]) -> float:
    m = max(xs)
    return m + math.log(sum(math.exp(x - m) for x in xs))


def renormalized_surprisals(summed: list[float]) -> list[float]:
    """Choice-level surprisal after renormalizing p(choice) over the choices."""
    lse = logsumexp([-s for s in summed])
    return [s + lse for s in summed]


def constrained_surprisals(tokens: list[list[str]], surprisals: list[list[float]]) -> list[list[float]]:
    """Per-token surprisal of each choice, renormalized over the choices' prefix tree.

    `tokens[j]` / `surprisals[j]`: the scored tokens of choice j and their full-vocabulary
    surprisals. Returns per-token constrained surprisals; sum them for the choice's surprisal.
    """
    for j, a in enumerate(tokens):
        if len(a) != len(surprisals[j]):
            raise ValueError(f"choice {j}: {len(a)} tokens but {len(surprisals[j])} surprisals")
        for k, b in enumerate(tokens):
            if j != k and len(a) < len(b) and b[:len(a)] == a:
                # would need p(end of answer | a), which per-token surprisals don't have
                raise ValueError(f"choice tokens {a} are a prefix of {b}; constrained scoring "
                                 "is not defined for this choice set")
    result = []
    for j, toks in enumerate(tokens):
        per_token = []
        for pos, tok in enumerate(toks):
            prefix = toks[:pos]
            allowed: dict[str, float] = {}  # next token -> its full-vocabulary surprisal
            for k, other in enumerate(tokens):
                if len(other) > pos and other[:pos] == prefix:
                    allowed.setdefault(other[pos], surprisals[k][pos])
            lse = logsumexp([-s for s in allowed.values()])
            # numerator read from the same entry as the denominator, so a forced token is exactly 0
            # (sequences sharing a prefix can differ slightly in bf16)
            per_token.append(allowed[tok] + lse)
        result.append(per_token)
    return result
