import tensorflow as tf
from skimage.transform import resize
import numpy as np
import os
import random
import matplotlib.pyplot as plt
import cv2


def normalise(matrix):
    """
    Normalizes a matrix by dividing it by its maximum value.

    Args:
        matrix (numpy.ndarray): The input matrix to be normalized.

    Returns:
        numpy.ndarray: The normalized matrix.
    """
    max_value = np.max(matrix)
    return matrix / max_value


def min_max_normalize(saliency_map):
    min_value = np.min(saliency_map)
    max_value = np.max(saliency_map)
    normalized_map = (saliency_map - min_value) / (max_value - min_value)
    return normalized_map


def scale_by_sum_normalize(saliency_map):
    sum_of_values = np.sum(saliency_map)
    normalized_map = saliency_map / sum_of_values
    return normalized_map


def apply_blur(image, sigma):
    # Applying Gaussian blur to the image
    blurred_image = cv2.GaussianBlur(image, (0, 0), sigma)
    return blurred_image


def get_last_layer_name(model_name):
    """
    Get the name of the last convolutional layer for a given model.

    Parameters:
    model_name (str): The name of the model, either 'Xception' or 'ResNet'.

    Returns:
    str: The name of the last convolutional layer.
    """
    if model_name == 'Xception':
        last_conv_layer_name = 'block14_sepconv2_act'
    elif model_name == 'ResNet':
        last_conv_layer_name = 'conv5_block3_out'
    return last_conv_layer_name


def get_image_index(filename):
    if filename.split('_')[0] == 'Colonial':
        index = 0
    elif filename.split('_')[0] == 'Modern':
        index = 1
    elif filename.split('_')[0] == 'Prehispanic':
        index = 2

    return index


def grid_layout(images, titles, suptitle):
    """
    Create and display a grid layout of images with titles.

    Args:
        images (list): List of images to be plotted.
        titles (list): List of titles for these images.
    """

    # Sets the title for the entire grid
    plt.suptitle('Grid layout of Results' + suptitle, fontsize=16)
    plt.figure(figsize=(12, 8))

    plt.subplot(221)
    im = plt.imshow(images[0])
    plt.colorbar(im, ax=plt.gca())
    plt.title(titles[0])
    plt.axis('off')

    plt.subplot(222)
    im = plt.imshow(images[1], cmap='gray')
    plt.colorbar(im, ax=plt.gca())
    plt.title(titles[1])
    plt.axis('off')

    plt.subplot(223)
    im = plt.imshow(images[2])
    plt.colorbar(im, ax=plt.gca())
    plt.title(titles[2])
    plt.axis('off')

    plt.subplot(224)
    im = plt.imshow(images[3])
    plt.colorbar(im, ax=plt.gca())
    plt.title(titles[3])
    plt.axis('off')
    # Adjusts the layout
    plt.tight_layout(pad=2.0)

    os.makedirs('report_output_images', exist_ok=True)
    save_index = random.randint(1, 100)
    plt.savefig('report_output_images/' + str(save_index) + '.jpg')

    # Shows the grid
    plt.show()


def plot_auc(deletion, n_values_d, scores_d, insertion, n_values_i, scores_i):
    plt.figure(figsize=(10, 10))

    plt.subplot(1, 2, 1)
    plt.title('Deletion AUC')
    plt.plot(n_values_d, scores_d)
    plt.xlabel('Percentage of deleted pixels')
    plt.ylabel('Score')
    plt.text(0.5, 0.9, f'AUC: {deletion:.3f}', transform=plt.gca(
    ).transAxes, fontsize=12, ha='center')

    plt.fill_between(n_values_d, 0, scores_d, alpha=0.2)

    plt.subplot(1, 2, 2)
    plt.title('Insertion AUC')
    plt.plot(n_values_i, scores_i)
    plt.xlabel('Percentage of inserted pixels')
    plt.ylabel('Score')
    plt.text(0.5, 0.9, f'AUC: {insertion:.3f}', transform=plt.gca(
    ).transAxes, fontsize=12, ha='center')
    plt.fill_between(n_values_i, 0, scores_i, alpha=0.2)
    plt.suptitle(f'AUC for Deletion and Insertion Evaluation', fontsize=16)

    plt.tight_layout()

    os.makedirs('report_output_images', exist_ok=True)
    save_index = random.randint(400, 600)
    plt.savefig('report_output_images/' + str(save_index) + '.jpg')
    plt.show()
