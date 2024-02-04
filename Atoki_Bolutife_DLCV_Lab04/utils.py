import tensorflow as tf
from skimage.transform import resize
import numpy as np
import os
import random
import matplotlib.pyplot as plt


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


def generate_masks(image, n_masks, mask_size, threshold):
    """
    Generates a set of random masks for an image.

    Args:
        image (numpy.ndarray): The input image.
        n_masks (int): The number of masks to generate.
        mask_size (int): The size of each mask.
        threshold (float): The threshold for mask value.

    Returns:
        list: A list containing perturbed images and corresponding masks.
    """

    H, W, _ = image.shape
    # resizing to the minimum dimension
    image = resize(image, (min(H, W), min(H, W)))

    # obtaining new dimensions
    H, W, _ = image.shape

    # upsampling mask dimensions using formular
    upsampled_H = int((mask_size + 1) * (H / mask_size))
    upsampled_W = int((mask_size + 1) * (W / mask_size))

    # Obtaining maximum displacement value for cropping origin point that doesn't exceed mask boundary
    diff_H = upsampled_H - H  # max position to move image inside upsampled mask
    diff_W = upsampled_W - W  # max position to move image inside upsampled mask

    # using threshold for deciding mask value; mask_value >= threshold --> 1
    masks = np.empty((n_masks, H, W))
    perturbed_images = np.empty((n_masks, H, W, 3))

    for i in range(n_masks):
        mask = (np.random.rand(mask_size, mask_size)
                >= threshold).astype('int')

        # random origin value that doesn't exceed upscaled mask size
        peturbed_x_origin = random.randint(0, diff_W)
        # random origin value that doesn't exceed upscaled mask size
        peturbed_y_origin = random.randint(0, diff_H)

        # Resizing mask and random cropping valid (same size as image) portion of mask
        masks[i, :, :] = resize(mask, (upsampled_H, upsampled_W), order=1,
                                mode='reflect', anti_aliasing=False)[peturbed_x_origin: peturbed_x_origin + W, peturbed_y_origin: peturbed_y_origin + H]

        # normalizing mask to 0 - 1 range
        masks[i, :, :] = normalise(masks[i, :, :])

        # Obtaining mask in 3 channels
        mask_3d = masks[i, :, :][..., None].repeat(3, axis=2)

        # blending mask and image
        pertubed_image = mask_3d * image

        perturbed_images[i, :, :, :] = pertubed_image

    return [perturbed_images, masks]


def make_prediction(model, model_name, perturbed_images, class_name):
    """
    Make predictions using a pre-trained deep learning model.

    Args:
        model (tensorflow.keras.models.Model): The pre-trained deep learning model.
        model_name (str): The name of the model ('Xception' or 'ResNet').
        perturbed_images (numpy.ndarray): An array of perturbed images.
        class_name (str): The target class name.

    Returns:
        tuple: A tuple containing scores and labels.
    """
    scores = []

    if model_name == 'Xception':
        preprocess_input = tf.keras.applications.xception.preprocess_input
        decode_predictions = tf.keras.applications.xception.decode_predictions
        size = (299, 299)

    elif model_name == 'ResNet':
        preprocess_input = tf.keras.applications.resnet_v2.preprocess_input
        decode_predictions = tf.keras.applications.resnet_v2.decode_predictions
        size = (224, 224)

    for image_index in range(perturbed_images.shape[0]):
        image = perturbed_images[image_index, :, :, :]

        # resizing image and expanding dimension
        image = resize(image, size)
        img_array = tf.expand_dims(preprocess_input(image), axis=0)

        # Performing prediction
        predictions = model.predict(img_array).flatten()

        # Getting the score for the class name
        labels = decode_predictions(np.asarray([predictions]), top=1000)[0]
        score = next(
            (label[2] for label in labels if label[1] == class_name)
        )

        scores.append(score)

    return scores, labels


def make_classifier(backbone):
    """
    Create a pre-trained deep learning classifier model.

    Args:
        backbone (str): The backbone model name ('Xception' or 'ResNet').

    Returns:
        tensorflow.keras.models.Model: The pre-trained deep learning classifier model.
    """

    if backbone == 'Xception':
        model_builder = tf.keras.applications.xception.Xception
        model = model_builder(weights="imagenet",
                              classifier_activation="softmax")

    elif backbone == 'ResNet':
        model_builder = tf.keras.applications.resnet_v2.ResNet50V2
        model = model_builder(weights="imagenet",
                              classifier_activation="softmax")

    return model


def calculate_saliency_map(scores, masks):
    """
    Calculate a saliency map based on scores and masks.

    Args:
        scores (list): List of scores.
        masks (numpy.ndarray): Array of masks.

    Returns:
        numpy.ndarray: The calculated saliency map.
    """
    sum_of_scores = np.sum(scores)
    saliency_map = np.zeros(masks[0].shape, dtype=np.float64)
    for i, mask_i in enumerate(masks):
        score_i = scores[i]
        saliency_map += score_i * mask_i

    saliency_map /= sum_of_scores
    return saliency_map


def grid_layout(images, titles):
    """
    Create and display a grid layout of images with titles.

    Args:
        images (list): List of images to be plotted.
        titles (list): List of titles for these images.
    """

    # Sets the title for the entire grid
    plt.suptitle('Grid layout of Results', fontsize=16)
    plt.figure(figsize=(12, 8))

    plt.subplot(121)
    plt.imshow(images[0])
    plt.title(titles[0])

    plt.subplot(122)
    plt.imshow(images[1])
    plt.title(titles[1])

    # # Adjusts the layout
    plt.tight_layout(pad=2.0)

    os.makedirs('output_images', exist_ok=True)
    save_index = random.randint(1, 100)
    plt.savefig('output_images/' + str(save_index) + '.jpg')

    # Shows the grid
    plt.show()
