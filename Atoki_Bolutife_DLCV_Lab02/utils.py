# utils.py
import numpy as np
from skimage.metrics import structural_similarity as ssim
from PIL import Image
import matplotlib.pyplot as plt
from constants import VERTICAL_RES, SCREEN_HEIGHT, SCREEN_WIDTH, A

def calculate_sigma(image, D, alpha):
    R = VERTICAL_RES / SCREEN_HEIGHT
    if (((SCREEN_WIDTH / image.size[1]) * image.size[0]) <= SCREEN_HEIGHT):
        ratio = image.size[1] / SCREEN_WIDTH
    else:
        ratio = image.size[0] / SCREEN_HEIGHT

    # return ratio * R * D * np.tan(alpha)
    return R * D * np.tan(alpha)

def calculate_error_metrics(ground_truth, saliency_map):
    ground_truth = np.array(ground_truth)
    saliency_map = np.array(saliency_map)

    mae = np.mean(np.abs(ground_truth - saliency_map))
    mse = np.mean((ground_truth - saliency_map) ** 2)
    ssim_score = ssim(ground_truth, saliency_map)
    pcc = np.corrcoef(ground_truth.flatten(), saliency_map.flatten())[0, 1]
    mean_saliency1 = np.mean(ground_truth)
    std_saliency1 = np.std(ground_truth)
    nss = np.mean((ground_truth - mean_saliency1) / std_saliency1)
    eps = 1e-10
    kl_divergence = np.sum(ground_truth * np.log((ground_truth + eps) / (saliency_map + eps)))
    metrics = [mae, mse, pcc, ssim_score]
    return metrics

def normalise(matrix):
    max_value = np.max(matrix)
    return matrix / max_value

def dict_to_txt(dictionary):
    file_path = "errors.txt"
    with open(file_path, "a") as file:
        for key, value in dictionary.items():
            file.write(f"{key}: {value}\n")

def calculate_partial_saliency_map(image, fixation_point, sigma):
    w, h = image.size
    x = np.arange(w)
    y = np.arange(h)

    X, Y = np.meshgrid(x, y)

    mesh_grid = np.dstack((X, Y))
    mesh_grid = (mesh_grid - fixation_point)**2 / (2 * sigma**2)
    mesh_grid = A * np.exp(-np.sum(mesh_grid, axis=2))
    return mesh_grid

def generate_saliency_map(image, fixation_points):
    sigma = calculate_sigma(image, D=325, alpha=np.deg2rad(2))
    saliency_map = 0.0
    for fixation_point in fixation_points:
        saliency_map += calculate_partial_saliency_map(image, fixation_point, sigma=sigma)
    saliency_map = normalise(saliency_map)
    return saliency_map
