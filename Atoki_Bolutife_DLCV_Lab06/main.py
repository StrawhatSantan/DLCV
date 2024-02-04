import os
import argparse
import cv2
from PIL import Image
from skimage.transform import resize
import tensorflow as tf

from GradCAM import *
from FEM import *
from RISE import *
from LIME import *

from representations import (represent_heatmap, represent_heatmap_overlaid
                             )

from constants import *
from utils import (normalise, get_image_index, grid_layout, plot_auc)
from evaluation import *


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--test_images_folder_path",
                        default='', help='Path to test images')
    parser.add_argument("--test_gdfm_folder_path",
                        default='', help='Path to test GDFMs')
    # parser.add_argument("--test_image_index",
    #                     default=0, help='Index can either be 0 for African Elephant or 1 for Black Bear')
    parser.add_argument("--explanation_method",
                        default='GRADCAM', help='Explanation type; GRADCAM, FEM, RISE, LIME')
    parser.add_argument("--model_name",
                        default='ResNet', help='Classifier: Either ResNet')
    parser.add_argument("--display_type", default='grid',
                        type=str, help='Layout to display results')

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    # Loading arguments
    images_folder_path = args.test_images_folder_path
    gdfm_folder_path = args.test_gdfm_folder_path
    # INDEX = int(args.test_image_index)
    DISPLAY_TYPE = args.display_type
    MODEL_NAME = args.model_name
    N_MASK = mask_number
    LOW_RES_MASK_SIZE = low_res_mask_size
    THRESHOLD = threshold
    METHOD = args.explanation_method
    colormap = 'jet'

    batch_errors = {'PCC': [],
                    'SSIM': [],
                    'Insertion': [],
                    'Deletion': []}

    for filename in os.listdir(images_folder_path):
        if filename == '.DS_Store':
            continue
        # Obtaining image, fixation, and GDFM names
        image_name = os.path.splitext(filename)[0]

        gdfm_filename = filename.replace("_N_", "_GFDM_N_")

        # obtaining paths to image, fixations, and GFDM's
        file_image_path = os.path.join(images_folder_path, filename)
        file_gdfm_path = os.path.join(gdfm_folder_path, gdfm_filename)

        # Loading image and Groundtruth GFDM
        image = tf.keras.preprocessing.image.load_img(file_image_path)
        gdfm_ground_truth = Image.open(file_gdfm_path).convert("L")

        INDEX = get_image_index(filename)

        # Preprocessing Image
        img_size = image.size
        image_array = tf.keras.preprocessing.image.img_to_array(image)
        img_del = np.transpose(image_array, (1, 0, 2))

        gdfm_ground_truth = np.transpose(np.array(gdfm_ground_truth))

        image_array = resize(image_array, size)
        image_array = tf.expand_dims(image_array, axis=0)

        model = tf.keras.models.load_model('model.h5', compile=False)

        if METHOD == 'GRADCAM':
            gradCAM = GradCAM(model, MODEL_NAME, image_array, INDEX)

            saliency_map = gradCAM.compute_saliency_map()

        elif METHOD == 'FEM':
            fem = FEM(model, MODEL_NAME, image_array)

            saliency_map = fem.compute_saliency_map()

        elif METHOD == 'RISE':
            Rise = RISE(model, image_array, INDEX, mask_number,
                        low_res_mask_size, threshold)

            saliency_map = Rise.compute_saliency_map()

        elif METHOD == 'LIME':
            saliency_map = explain_with_lime(model, image_array,
                                             top_labels, hide_color, num_lime_features, num_samples,
                                             positive_only, negative_only, num_superpixels, hide_rest, rand_index)

            # Define the standard deviation (sigma) for the Gaussian blur
            sigma = 0.5

            # Apply Gaussian blur to the heatmap
            saliency_map = cv2.GaussianBlur(saliency_map, (0, 0), sigma)

        saliency_map = normalise(saliency_map)

        # Resizing saliency map to input image size
        saliency_map = resize(saliency_map, img_size, order=3,
                              mode='wrap', anti_aliasing=False)

        # saliency_map = normalise(saliency_map)
        gray_saliency = represent_heatmap(saliency_map, 'gray')

        # Representing Saliency map with heatmap
        heatmapped_saliency = represent_heatmap(saliency_map, colormap)
        heatmapped_saliency = heatmapped_saliency.resize(img_size)

        # Overlaying saliency map on input image
        saliency_blended = represent_heatmap_overlaid(
            saliency_map, image, colormap)

        # Obtaining Error metrics for evaluating saliency map
        pcc = calculate_pcc(normalise(gdfm_ground_truth), saliency_map)
        ssim = calculate_ssim(normalise(gdfm_ground_truth), saliency_map)

        ins_auc, ins_scores, n_values_i = insertion(
            model, np.array(tf.squeeze(image_array)), resize(saliency_map, size), 500, INDEX)

        del_auc, del_scores, n_values_d = deletion(
            model, np.array(tf.squeeze(image_array)), resize(saliency_map, size), 500, INDEX)

        error_names = ['PCC', 'SSIM', 'Insertion', 'Deletion']
        errors = [pcc, ssim, ins_auc, del_auc]

        for i, error in enumerate(range(len(errors))):
            batch_errors[error_names[i]].append(errors[i])

        # For plotting grid layout of images
        images = [gray_saliency, gdfm_ground_truth,
                  heatmapped_saliency, saliency_blended]
        titles = ['Saliency map', 'Groundtruth Saliency Map',
                  'Heat Mapped Saliency Image', 'Blended Saliency Image']
        supertitle = str(METHOD) + 'Explanation Method' + \
            ', using' + str(MODEL_NAME) + 'Transfer Learned Model'

        plot_auc(del_auc, n_values_d, del_scores,
                 ins_auc, n_values_i, ins_scores)

        # Deciding display type
        if DISPLAY_TYPE == 'singles':
            for image in images:
                image.show()

        elif DISPLAY_TYPE == 'grid':
            grid_layout(images, titles, supertitle)

    # printing batch results obtained
    for i, error in enumerate(range(len(batch_errors))):
        batch_errors[error_names[i]] = error_names[i] + '---> mean: ' + str(np.mean(
            batch_errors[error_names[i]])) + ', variance: ' + str(np.std(batch_errors[error_names[i]]))
        print(batch_errors[error_names[i]])
