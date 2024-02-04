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


def get_model(model_name):
    """
    Create a model for a specified architecture and obtain the name of the last layer.

    Parameters:
    model_name (str): The name of the model, either 'Xception' or 'ResNet'.

    Returns:
    (tf.keras.Model, str): A tuple containing the model and the name of the last convolutional layer.
    """
    model = make_classifier(model_name)
    model.layers[-1].activation = None  # Removing last layer's softmax
    last_layer_name = get_last_layer_name(
        model_name)  # Getting last layer's name
    return model, last_layer_name


def get_decode_predictions(model_name):
    """
    Get the function for decoding model predictions for a specified architecture.

    Parameters:
    model_name (str): The name of the model, either 'Xception' or 'ResNet'.

    Returns:
    Callable: A function for decoding model predictions.
    """
    if model_name == 'Xception':
        decode_predictions = tf.keras.applications.xception.decode_predictions
    elif model_name == 'ResNet':
        decode_predictions = tf.keras.applications.resnet_v2.decode_predictions
    return decode_predictions


def get_preprocess_input(model_name):
    """
    Get the preprocess_input function for a specified architecture.

    Parameters:
    model_name (str): The name of the model, either 'Xception' or 'ResNet'.

    Returns:
    Callable: A function for preprocessing input images.
    """
    if model_name == 'Xception':
        preprocess_input = tf.keras.applications.xception.preprocess_input
    elif model_name == 'ResNet':
        preprocess_input = tf.keras.applications.resnet_v2.preprocess_input
    return preprocess_input


def get_required_size(model_name):
    """
    Get the required input image size for a specified architecture.

    Parameters:
    model_name (str): The name of the model, either 'Xception' or 'ResNet'.

    Returns:
    tuple: A tuple containing the required image size as (width, height).
    """
    if model_name == 'Xception':
        size = (299, 299)
    elif model_name == 'ResNet':
        size = (224, 224)
    return size


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
