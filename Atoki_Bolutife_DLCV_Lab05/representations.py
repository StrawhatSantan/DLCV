from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from utils import normalise


def represent_heatmap(saliency, cmap='gray'):
    """"
    Inputs:
    Saliency map as image (either unsigned or signed),
    Colormap to be used

    Ouput:
    Heatmapped saliency map 
    """
    if np.max(saliency) > 1:
        saliency = normalise(saliency)

    colormap = plt.get_cmap(cmap)
    heatmapped_saliency = (colormap(saliency) * 255).astype(np.uint8)

    heatmapped_saliency_image = Image.fromarray(heatmapped_saliency)
    return heatmapped_saliency_image


def represent_heatmap_overlaid(saliency, image, cmap):
    """
    Inputs:
    Saliency map as image (either unsigned or signed),
    RGB Image,
    Colormap

    Output:
    RGBImage overlaid with saliency map
    """
    # Calling represent_heatmap to generate heatmapped saliency using specified
    heatmapped_saliency = represent_heatmap(saliency, cmap)
    heatmapped_saliency = heatmapped_saliency.resize(image.size)

    # Blending Input Image with saliency heatmap image
    blended_image = Image.blend(image.convert(
        'RGBA'), heatmapped_saliency.convert('RGBA'),  alpha=0.5)

    return blended_image
