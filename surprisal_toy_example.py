import seaborn as sns
import matplotlib.pyplot as plt
from tools.model_of_language import *
from utils.color_palette import *

# sentence = Sentence(
#     context='Mary went to her office. It is ', 
#     target=' that Mary works.', 
#     intervened_target=' that Mary works.',
#     interventions=[
#         # Intervention(text="he had displayed genuine strokes of boldness two years ago"),
#         Intervention(text="very atypical"),
#         Intervention(text="somewhat atypical"),
#         Intervention(text="neither typical nor atypical"),
#         Intervention(text="somewhat typical"),
#         Intervention(text="very typical"),
#     ]
# )
sentence = Sentence(
    context='Mary went to her office. She worked. It is ', 
    target=' that Mary works.', 
    intervened_target=' that Mary works.',
    interventions=[
        # Intervention(text="he had displayed genuine strokes of boldness two years ago"),
        Intervention(text="very atypical"),
        Intervention(text="somewhat atypical"),
        Intervention(text="neither typical nor atypical"),
        Intervention(text="somewhat typical"),
        Intervention(text="very typical"),
    ]
)

mol = ModelOfLanguage(path='/public/hf/models/mistralai/Mistral-7B-v0.1', nickname="Mistral7B")

print(sentence.pretty())
sentence = mol.get_measures_for_sentence(sentence)

# print("HEY")
# sns.set_style("whitegrid")
# sns.set_context("paper")
# # Set the custom palette
# sns.set_palette(custom_palette)
for i, intervention in enumerate(sentence.interventions):
   print(f"intervention: {intervention.text}")
   print(intervention.measures["surprisal"])
   print(sentence.measure_tokens)
   print('\n')


# def plot_sentence(sentence, save_path=None, measure_name="surprisal"):
#     fig, ax = plt.subplots(figsize=(7, 4))
#     sns.lineplot(
#             y=sentence.target_measures[measure_name],
#             x=sentence.measure_tokens, 
#             marker='X', 
#             linestyle='-', 
#             linewidth=2, 
#             label=f"GEN",
#             color=custom_palette[0],  # Use custom colors
#             ax=ax
#         )
#     ax.set_xticklabels(ax.get_xticklabels())
#     ax.set_yticks([0,5,10])
#     # add a horizontal line at the mean of the target measures
#     for i, intervention in enumerate(sentence.interventions):
#         sns.lineplot(
#             y=intervention.measures[measure_name], 
#             x=sentence.measure_tokens, 
#             marker='o' if i == 0 else 's' if i == 1 else '^', 
#             linestyle='-', 
#             linewidth=2, 
#             label=f"{intervention.text}",
#             color=custom_palette[i+1],  # Use custom colors
#             ax=ax
#         )
#     ax.set_xticklabels([r"{}".format(tick.get_text()) for tick in ax.get_xticklabels()], size=10)
#     # also bottom
#     sns.despine(bottom=True)
#     # generic

#     plt.xlabel("")
#     plt.ylabel(r"{}".format(measure_name.capitalize()))
#     plt.title("")
#     legend_labels = [r"Generic"] + [r"{}".format(intervention.text) for intervention in sentence.interventions]
#     legend_handles = []
#     # legend handles are small lineplots
#     # from matplotlib.lines import Line2D

#     # handles = [
#     #     Line2D([0], [0], color=custom_palette[0], lw=2, ls="-", marker="X",  markeredgecolor="white",),
#     #     Line2D([0], [0], color=custom_palette[1], lw=2, ls="-", marker="o",  markeredgecolor="white",),
#     #     Line2D([0], [0], color=custom_palette[2], lw=2, ls="-", marker="s", markeredgecolor="white",),
#     #     Line2D([0], [0], color=custom_palette[4], lw=2, ls="-", marker="^", markeredgecolor="white",),
#     # ]

#     # plt.legend(
#     #     handles=handles,
#     #     labels=legend_labels,
#     #     fontsize=9,
#     #     # set labels as \textrm
#     #     # loc="upper right"
#     #     loc="center left",
#     #     bbox_to_anchor=(1.02, 0.5),
#     #     borderaxespad=0.
#     # )
#     plt.grid(
#         True,
#         linestyle='--',
#         linewidth=0.5,
#         alpha=0.3,
#         color='lightgray'
#     )
#     plt.tight_layout()
#     if save_path:
#         plt.savefig(save_path, format=save_path.split(".")[-1], bbox_inches='tight', dpi=300)
#     plt.show()
# plot_sentence(sentence, measure_name="surprisal")