import cv2
import numpy as np
from utils import min_max_normalize, scale_by_sum_normalize
from skimage.metrics import structural_similarity as ssim
import tensorflow as tf


def calculate_auc(x, y):
    return np.trapz(y, x)


def set_n_pixels_deletion(image_array, saliency_map, n_pixels):
    flattened_saliency = saliency_map.flatten()
    sorted_indices = np.argsort(flattened_saliency)[::-1][:n_pixels]
    flattened_saliency[sorted_indices] = 0
    modified_saliency = flattened_saliency.reshape(saliency_map.shape)

    row_indices, col_indices = np.unravel_index(
        sorted_indices, saliency_map.shape)
    modified_image = np.squeeze(image_array)

    modified_image[row_indices, col_indices, :] = 0
    modified_image = modified_image.reshape(1, *modified_image.shape)
    return modified_image, modified_saliency


def predict_scores(model, image, class_index):
    image = np.expand_dims(image, axis=0)
    predictions = model.predict(image).flatten()
    class_probabilities = np.exp(
        predictions) / np.sum(np.exp(predictions), axis=-1, keepdims=True)
    print(class_probabilities)
    image = np.squeeze(image)
    score = class_probabilities[class_index]
    print("Score:", score)

    return score


def deletion(model, image_array, saliency_map, n_pixels, class_index):
    """
    Calculate deletion score for an img_array using the DELETION algorithm.

    Parameters:
    model (callable): A blackbox model that takes an img_array as input and returns a prediction.
    img_array (numpy.ndarray): The input img_array.
    importance_map (numpy.ndarray): The saliency / explanation map for the img_array.
    num_pixels_to_remove (int): Number of pixels to remove per step.

    Returns:
    deletion_score (float): The deletion score.
    """

    scores = []
    image_array1 = image_array.copy()
    image_slice = np.squeeze(image_array1)[:, :, 0]
    n = 0

    score = predict_scores(model, image_array1, class_index)
    scores.append(score)
    while np.any(image_slice != 0):
        image_array1, saliency_map = set_n_pixels_deletion(
            image_array1, saliency_map, n_pixels
        )
        image_slice = np.squeeze(image_array1)[:, :, 0]
        n = n + 1
        image_array1 = np.squeeze(image_array1)
        score = predict_scores(model, image_array1, class_index)
        scores.append(score)
        if n > n_pixels:
            break

    n_values = [index / n for index in range(n + 1)]
    score_d = calculate_auc(n_values, scores)

    return score_d, scores, n_values


def set_n_pixels_insertion(image_array, blurred_image, saliency_map, n_pixels):
    flattened_saliency = saliency_map.flatten()
    sorted_indices = np.argsort(flattened_saliency)[::-1][:n_pixels]
    flattened_saliency[sorted_indices] = 0
    modified_saliency = flattened_saliency.reshape(saliency_map.shape)

    row_indices, col_indices = np.unravel_index(
        sorted_indices, saliency_map.shape)
    modified_blurred_image = np.squeeze(blurred_image)
    image_array = np.squeeze(image_array)
    modified_blurred_image[row_indices, col_indices, :] = image_array[
        row_indices, col_indices, :
    ]
    modified_blurred_image = modified_blurred_image.reshape(
        1, *modified_blurred_image.shape
    )
    return modified_blurred_image, modified_saliency


def insertion(model, image_array, saliency_map, n_pixels, class_index):
    scores = []
    image_slice = np.squeeze(image_array)[:, :, 0]
    blurred_image_array = cv2.GaussianBlur(image_array, (127, 127), 0)
    blurred_image_slice = blurred_image_array[:, :, 0]
    n = 0
    score = 0
    scores.append(score)
    while np.any(blurred_image_slice != image_slice):
        blurred_image_array, saliency_map = set_n_pixels_insertion(
            image_array, blurred_image_array, saliency_map, n_pixels
        )
        blurred_image_slice = np.squeeze(blurred_image_array)[:, :, 0]
        n = n + 1
        blurred_image_array = np.squeeze(blurred_image_array)
        score = predict_scores(model, blurred_image_array, class_index)
        scores.append(score)

        if n > n_pixels:
            break

    n_values = [index / n for index in range(n + 1)]
    score_i = calculate_auc(n_values, scores)

    return score_i, scores, n_values


def calculate_sim(calculated_saliency, gt_saliency):
    gt_saliency = min_max_normalize(gt_saliency)

    calculated_saliency = scale_by_sum_normalize(calculated_saliency)
    gt_saliency = scale_by_sum_normalize(gt_saliency)

    sim_value = np.sum(np.minimum(gt_saliency, calculated_saliency))

    return sim_value


def calculate_pcc(ground_truth, saliency_map):

    pcc = np.corrcoef(ground_truth.flatten(), saliency_map.flatten())[0, 1]

    return pcc
