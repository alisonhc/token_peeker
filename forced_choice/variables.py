# Answer choices per scale, lowest to highest. Same option sets as the instruct scales in
# llm-structured-inference/utils/variables.py. The scale must match the question in the
# input prompt ("how likely/typical/probable/often is it that ...").
SCALES: dict[str, list[str]] = {
    "likely": ["Very Unlikely", "Unlikely", "Likely", "Very Likely"],
    "typical": ["Very Atypical", "Atypical", "Typical", "Very Typical"],
    "probable": ["Very Improbable", "Improbable", "Probable", "Very Probable"],
    "often": ["Very Rarely", "Rarely", "Often", "Very Often"],
}
DEFAULT_SCALE = "likely"


def get_answer_choices(scale: str = DEFAULT_SCALE) -> list[str]:
    if scale not in SCALES:
        raise ValueError(f"Unknown scale {scale!r}; choose from {list(SCALES)}")
    return SCALES[scale]


def get_answer_choice_dict(scale: str = DEFAULT_SCALE) -> dict[str, int]:
    # same numbering as tools/variables.py, so the output columns line up with the
    # surprisal_script output (1 = lowest ... 4 = highest)
    return {choice: i for i, choice in enumerate(get_answer_choices(scale), start=1)}


def get_options_instruction(scale: str = DEFAULT_SCALE) -> str:
    # appended to the end of every prompt, before the answer choice
    return f"Choose one of the following options: {', '.join(get_answer_choices(scale))}.\n"


# Defaults ("likely" scale), kept for backwards compatibility.
ANSWER_CHOICES = get_answer_choices(DEFAULT_SCALE)
ANSWER_CHOICE_DICT = get_answer_choice_dict(DEFAULT_SCALE)
OPTIONS_INSTRUCTION = get_options_instruction(DEFAULT_SCALE)
VERY_UNLIKELY, UNLIKELY, LIKELY, VERY_LIKELY = ANSWER_CHOICES

# what goes between the prompt from the input file and the options instruction
PROMPT_JOINER = " "

# Used for every instruction-tuned model (i.e. with --chat-template), so that the system
# message is held constant across models instead of each chat template supplying its own
# default. Pass --system-prompt "" to leave it out entirely.
SYSTEM_PROMPT = "You are a helpful assistant."

# what goes between the options instruction and the answer choice being scored.
# The options instruction already ends in a newline, so nothing extra by default.
CHOICE_JOINER = ""
