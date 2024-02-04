import argparse
from PIL import Image
from skimage.transform import resize
import tensorflow as tf

from representations import (represent_heatmap, represent_heatmap_overlaid
                             )
from utils import (generate_masks, make_classifier, make_prediction,
                   calculate_saliency_map, grid_layout)

from constants import (classes, low_res_mask_size, mask_number)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--test_image_path",
                        default='test.jpg', help='Path to test image')
    parser.add_argument("--test_image_index",
                        default=0, help='Index can either be 0 for African Elephant ot 1 for Black Bear')
    parser.add_argument("--model_name",
                        default='ResNet', help='Classifier: Either ResNet or Xception')
    parser.add_argument("--display_type", default='grid',
                        type=str, help='Layout to display results')

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    # Loading arguments
    image_path = args.test_image_path
    INDEX = int(args.test_image_index)
    DISPLAY_TYPE = args.display_type
    MODEL_NAME = args.model_name
    N_MASK = mask_number
    LOW_RES_MASK_SIZE = low_res_mask_size
    colormap = 'turbo'
    class_name = classes[INDEX]

    # Loading test image
    image = Image.open(image_path)
    img_size = image.size

    # Getting array of image
    image_array = tf.keras.preprocessing.image.img_to_array(image)

    # Obtaining masks for test image
    perturbed_images, masks = generate_masks(
        image_array, N_MASK, LOW_RES_MASK_SIZE, 0.3)

    # Loading Model
    model = make_classifier(MODEL_NAME)

    # Obtaining scores
    scores, _ = make_prediction(
        model, MODEL_NAME, perturbed_images, class_name)

    # Obtaining saliency map
    saliency_map = calculate_saliency_map(scores, masks)
    saliency_map = resize(saliency_map, img_size)

    # Representing Saliency map with heatmap
    heatmapped_saliency = represent_heatmap(saliency_map, colormap)
    heatmapped_saliency = heatmapped_saliency.resize(image.size)

    # Overlaying saliency map on input image
    saliency_blended = represent_heatmap_overlaid(
        saliency_map, image, colormap)

    # For plotting grid layout of images
    images = [heatmapped_saliency, saliency_blended]
    titles = ['Heat Mapped Saliency Image', 'Blended Saliency Image', 'Perturbed Image Sample',
              'Binary Mask Sample']

    # Deciding display type
    if DISPLAY_TYPE == 'singles':
        for image in images:
            image.show()

    elif DISPLAY_TYPE == 'grid':
        grid_layout(images, titles)
