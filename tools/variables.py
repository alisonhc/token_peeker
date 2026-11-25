# Intervention scales for the surprisal experiments (base models).
# Each scale is inserted between the context ("... it is ") and the target, e.g.
#   "In general, when X, it is " + "very unlikely that" + " Y happens."
# Wording mirrors the instruct prompts ("how likely/typical/probable/often is it that ...")
# and the scales in llm-structured-inference/utils/variables.py.
# Keys are 1 = lowest ... 4 = highest, matching the output columns ("1_surprisal", ...).

SCALES: dict[str, list[str]] = {
    "likely": ["very unlikely that", "unlikely that", "likely that", "very likely that"],
    "typical": ["very atypical that", "atypical that", "typical that", "very typical that"],
    "probable": ["very improbable that", "improbable that", "probable that", "very probable that"],
    # Bare adverbs, no "it is ... that": use with the *_forbasemodels_often.jsonl inputs, whose
    # contexts have "it is " removed (llm-structured-inference/data_processing/make_often_surprisal_input.py),
    # e.g. "In general, when X, " + "very often" + " Y happens."
    "often": ["very rarely", "rarely", "often", "very often"],
}
DEFAULT_SCALE = "likely"


def get_interventions(scale: str = DEFAULT_SCALE) -> list[str]:
    if scale not in SCALES:
        raise ValueError(f"Unknown intervention scale {scale!r}; choose from {list(SCALES)}")
    return SCALES[scale]


def get_intervention_dict(scale: str = DEFAULT_SCALE) -> dict[str, int]:
    return {text: i for i, text in enumerate(get_interventions(scale), start=1)}


# Kept for backwards compatibility (default "likely" scale).
INTERVENTIONS = get_interventions(DEFAULT_SCALE)
INTERVENTION_DICT = get_intervention_dict(DEFAULT_SCALE)
VERY_UNLIKELY, UNLIKELY, LIKELY, VERY_LIKELY = INTERVENTIONS
