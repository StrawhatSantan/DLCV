import argparse
from PIL import Image
from skimage.transform import resize
import tensorflow as tf
import cv2

from GRADCAM import *
from FEM import *
from representations import (represent_heatmap, represent_heatmap_overlaid
                             )
from constants import (classes, low_res_mask_size, mask_number)
from utils import normalise, get_model, get_decode_predictions, get_preprocess_input, get_required_size, grid_layout


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--test_image_path",
                        default='test.jpg', help='Path to test image')
    parser.add_argument("--test_image_index",
                        default=0, help='Index can either be 0 for African Elephant or 1 for Black Bear')
    parser.add_argument("--explanation_method",
                        default='GRADCAM', help='Explanation type; GRADCAM or FEM')
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
    METHOD = args.explanation_method
    colormap = 'turbo'
    class_name = classes[INDEX]

    # Loading image
    image = Image.open(image_path)
    img_size = image.size

    # Getting array of image
    image_array = tf.keras.preprocessing.image.img_to_array(image)

    if METHOD == 'GRADCAM':

        # Initializing GRADCAM Object
        Grad_cam = GradCAM(MODEL_NAME, image_array)

        # Obtaining GRADCAM model
        grad_cam_model = Grad_cam.get_model()

        # obtaining gradients of last conv layer wrt to prediction of class of input image
        gradients = Grad_cam.compute_gradients(grad_cam_model, class_name)

        # Performing global average pooling of gradients to get 1x1 scalar
        pooled_gradients = Grad_cam.pool_gradients(gradients)

        # weighting each activation map by its corresponding scalar value
        Grad_cam.weight_activation_map(pooled_gradients)

        # Setting negative values in weighted maps to 0
        Grad_cam.apply_relu()

        # Performing dimension pooling to get 1 channel array to get saliency map
        saliency_map = Grad_cam.apply_dimension_average_pooling()

        # normalizing saliency map
        saliency_map = normalise(saliency_map)

        # smoothening heatmap
        # Define the standard deviation (sigma) for the Gaussian blur
        sigma = 0.5

        # Apply Gaussian blur to the heatmap
        saliency_map = cv2.GaussianBlur(saliency_map, (0, 0), sigma)

    elif METHOD == 'FEM':
        # Making model
        model, last_conv_layer_name = get_model(MODEL_NAME)

        # Importing process functions
        preprocess_input = get_preprocess_input(MODEL_NAME)
        decode_predictions = get_decode_predictions(MODEL_NAME)

        # Preparing image
        img_array = tf.expand_dims(preprocess_input(
            resize(image_array, get_required_size(MODEL_NAME))), axis=0)

        # Generating saliency with FEM algorithm
        saliency_map = compute_fem(img_array, model, last_conv_layer_name)

    saliency_map = normalise(saliency_map)

    # Resizing saliency map to input image size
    saliency_map = resize(saliency_map, img_size, order=3,
                          mode='wrap', anti_aliasing=False)

    # Representing Saliency map with heatmap
    heatmapped_saliency = represent_heatmap(saliency_map, colormap)
    heatmapped_saliency = heatmapped_saliency.resize(image.size)

    # Overlaying saliency map on input image
    saliency_blended = represent_heatmap_overlaid(
        saliency_map, image, colormap)

    # For plotting grid layout of images
    images = [heatmapped_saliency, saliency_blended]
    titles = ['Heat Mapped Saliency Image', 'Blended Saliency Image']
    supertitle = str(METHOD) + 'Method' + ',' + str(MODEL_NAME) + 'Model'

    # Deciding display type
    if DISPLAY_TYPE == 'singles':
        for image in images:
            image.show()

    elif DISPLAY_TYPE == 'grid':
        grid_layout(images, titles, supertitle)
