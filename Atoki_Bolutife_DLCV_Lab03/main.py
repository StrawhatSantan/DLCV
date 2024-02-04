import os
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from representations import (normalise, represent_heatmap, represent_heatmap_overlaid, represent_isolines,
                             represent_isolines_superimposed, represent_hard_selection, represent_soft_selection)
from objects import (Saliency, RGBImage)
from utils import grid_layout

# Arguments Parser Declaration


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_image_path",
                        default='test.jpg', help='Path to test image')
    parser.add_argument("--test_saliency_map_path",
                        default='test_saliency_img.png', help='Path to test saliency map')
    parser.add_argument("--display_type", default='grid',
                        type=str, help='Layout to display results')

    return parser.parse_args()

# python main.py --test_image_path='/Users/topboy/IPCV/UBx/DLCV/Lab03/MexCulture142/images_train/Colonial_CatedralDeVeracruzNuestraSenoraDeLaAsuncion_Veracruz_N_1.png' --test_saliency_map_path='/Users/topboy/IPCV/UBx/DLCV/Lab03/MexCulture142/gazefixationsdensitymaps/Colonial_CatedralDeVeracruzNuestraSenoraDeLaAsuncion_Veracruz_GFDM_N_1.png' --display_type='grid'


if __name__ == "__main__":

    args = parse_args()

    test_image_path = args.test_image_path
    test_saliency_map_path = args.test_saliency_map_path
    # fixations_folder_path = args.fixations_path
    # fixations_maps_ground_truths = args.gt_path
    display_type = args.display_type

    # image = Image.open('test.jpg')
    # saliency = Image.open('test_saliency_img.png')

    image = Image.open(test_image_path)
    saliency = Image.open(test_saliency_map_path)

    saliency = np.array(saliency)

    Saliency = Saliency(saliency)
    saliency = Saliency.image

    Image = RGBImage(image)
    image = Image.image

    colormap = 'gist_heat'

    # Obtaining all required images
    heatmapped_saliency = represent_heatmap(saliency, colormap)
    saliency_blended = represent_heatmap_overlaid(saliency, image, colormap)
    heatmapped_isolines = represent_isolines(saliency,  colormap)
    isoline_blended = represent_isolines_superimposed(
        saliency, image, colormap)
    hard_image = represent_hard_selection(saliency, image, 200)
    soft_image = represent_soft_selection(saliency, image)

    # For plotting grid layout of images
    images = [heatmapped_saliency, saliency_blended,
              heatmapped_isolines, isoline_blended, hard_image, soft_image]
    titles = ['Heat Mapped Saliency Image', 'Blended Saliency Image', 'Heat Mapped Isoline Image',
              'Blended Isoline Image', 'Hard Masked Image', 'Soft Masked Image']

    # Deciding display type
    if display_type == 'singles':
        for image in images:
            image.show()

    elif display_type == 'grid':
        grid_layout(images, titles)
