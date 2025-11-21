import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


colors = {
    "colors":[
        {"value":"rgba(12, 187, 39, 1)","a":1},
        
        
        {"value":"rgba(217, 31, 149, 1)","a":1},
        {"value":"rgba(47, 42, 212, 1)","a":1},
        
        
        {"value":"rgba(31, 176, 202, 1)","a":1}, 
        {"value":"rgba(232, 225, 31, 1)","a":1},
        
        {"value":"rgba(71, 88, 106, 1)","a":1},
        
        {"value":"rgba(17, 18, 23, 1)","a":1},
        
        {"value":"rgba(16, 103, 35, 1)","a":1},
        
        {"value":"rgba(209, 140, 71, 1)","a":1},
        {"value":"rgba(238, 218, 231, 1)","a":1},
        {"value":"rgba(215, 68, 44, 1)","a":1},
        {"value":"rgba(136, 221, 30, 1)","a":1},
        # {"value":"rgba(238, 218, 231, 1)","a":1},
    ]
}#{"value":"rgba(238, 218, 231, 1)","a":1},
# colors = {"colors":[{"value":"rgba(225, 229, 200, 1)","a":1},{"value":"rgba(9, 20, 40, 1)","a":1},{"value":"rgba(215, 68, 44, 1)","a":1},{"value":"rgba(144, 159, 115, 1)","a":1},{"value":"rgba(127, 88, 66, 1)","a":1},{"value":"rgba(200, 133, 74, 1)","a":1},{"value":"rgba(65, 99, 67, 1)","a":1},{"value":"rgba(71, 88, 106, 1)","a":1},{"value":"rgba(45, 59, 81, 1)","a":1},{"value":"rgba(118, 131, 158, 1)","a":1}]}

# convert into sns color palette
rgb_values = [(int(color["value"].split(',')[0].split('(')[1].strip())/255.0, 
               int(color["value"].split(',')[1].strip())/255.0, 
               int(color["value"].split(',')[2].strip())/255.0) for color in colors["colors"]]

# Convert RGB values to Seaborn color palette
custom_palette = sns.color_palette(rgb_values, n_colors=len(rgb_values))
# tnr = FontProperties(family='Times New Roman', style='normal', size=12)
# plt.style.use(['science'])
# update rdparams with custom ttf
# plt.rcParams.update({
#     "font.family": "serif",   # specify font family here
#     "font.serif": ["Times New Roman"],  # specify font here
#     "font.size":11
#     })

# plt.rcParams.update({
#     "font.family": "serif",   # specify font family here
#     "font.serif": ["Times New Roman"],  # specify font here
#     "font.size":11
#     })    
from colorsys import rgb_to_hls, hls_to_rgb

def adjust_color(color, lightness_factor=0.9, saturation_factor=0.8):
    """
    Adjust the color to be darker and duller.
    
    :param color: Original RGB color tuple.
    :param lightness_factor: Factor to adjust lightness (default: 0.8).
    :param saturation_factor: Factor to adjust saturation (default: 0.8).
    :return: Adjusted RGB color tuple.
    """
    # Convert RGB to HLS
    h, l, s = rgb_to_hls(*color)
    
    # Adjust lightness and saturation
    l_adjusted = max(0, min(l * lightness_factor, 1))  # Darken color
    s_adjusted = max(0, min(s * saturation_factor, 1))  # Make color duller
    
    # Convert HLS back to RGB
    adjusted_color = hls_to_rgb(h, l_adjusted, s_adjusted)
    
    return adjusted_color

# Example usage
custom_palette = [adjust_color(color) for color in custom_palette]

sns.palplot(custom_palette) 