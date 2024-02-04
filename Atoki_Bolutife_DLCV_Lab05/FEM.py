import numpy as np
import tensorflow as tf
import functools
import keract

# Display
# from IPython.display import Image, display
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def get_img_array(img_path, size):
    """
    Load an image from a file and convert it to a numpy array.

    Parameters:
    img_path (str): The path to the image file.
    size (tuple): The target size for the image (width, height).

    Returns:
    np.ndarray: A numpy array representing the image.
    """
    # `img` is a PIL image of size 299x299
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=size)
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = tf.keras.preprocessing.image.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)
    return array


def expand_flat_values_to_activation_shape(values, W_layer, H_layer):
    """
    Expand a 1D array of values to the shape of a neural network activation map.

    Parameters:
    values (np.ndarray): 1D array of values to be expanded.
    W_layer (int): Width of the activation map.
    H_layer (int): Height of the activation map.

    Returns:
    np.ndarray: An expanded array with the shape (W_layer, H_layer, len(values)).
    """
    if False:
        # Initial implementation in original FEM paper
        expanded = np.expand_dims(values, axis=1)
        expanded = np.kron(expanded, np.ones((W_layer, 1, H_layer)))
        expanded = np.transpose(expanded, axes=[0, 2, 1])
    else:
        # Simplified implementation
        expanded = values.reshape((1, 1, -1)) * \
            np.ones((W_layer, H_layer, len(values)))
    return expanded


def compute_binary_maps(feature_map, sigma=None):
    """
    Compute binary maps based on feature maps.

    Parameters:
    feature_map (np.ndarray): The feature map to be thresholded.
    sigma (float): The standard deviation for thresholding (default is 2).

    Returns:
    np.ndarray: Binary maps based on feature map values.
    """

    batch_size, W_layer, H_layer, N_channels = feature_map.shape
    thresholded_tensor = np.zeros((batch_size, W_layer, H_layer, N_channels))

    if sigma is None:
        feature_sigma = 2
    else:
        feature_sigma = sigma

    for B in range(batch_size):
        # Get the activation value of the current sample
        activation = feature_map[B, :, :, :]

        # Calculate its mean and its std per channel
        mean_activation_per_channel = activation.mean(axis=(0, 1))
        std_activation_per_channel = activation.std(axis=(0, 1))
        assert len(mean_activation_per_channel) == N_channels
        assert len(std_activation_per_channel) == N_channels

        # Transform the mean in the same shape than the activation maps
        mean_activation_expanded = expand_flat_values_to_activation_shape(
            mean_activation_per_channel, W_layer, H_layer)

        # Transform the std in the same shape than the activation maps
        std_activation_expanded = expand_flat_values_to_activation_shape(
            std_activation_per_channel, W_layer, H_layer)

        # Build the binary map
        thresholded_tensor[B, :, :, :] = 1.0 * (activation > (
            mean_activation_expanded + feature_sigma * std_activation_expanded))

    return thresholded_tensor


def aggregate_binary_maps(binary_feature_map, orginal_feature_map):
    """
    Aggregate binary maps using the original feature map.

    Parameters:
    binary_feature_map (np.ndarray): Binary maps.
    orginal_feature_map (np.ndarray): Original feature map.

    Returns:
    np.ndarray: Aggregated feature map.
    """

    # This weigths the binary map based on original feature map
    batch_size, W_layer, H_layer, N_channels = orginal_feature_map.shape

    orginal_feature_map = orginal_feature_map[0]
    binary_feature_map = binary_feature_map[0]

    # Get the weights
    # Take means for each channel-values
    channel_weights = np.mean(orginal_feature_map, axis=(0, 1))
    if False:
        # Original paper implementation
        expanded_weights = np.kron(np.ones(
            (binary_feature_map.shape[0], binary_feature_map.shape[1], 1)), channel_weights)
    else:
        # Simplified version
        expanded_weights = expand_flat_values_to_activation_shape(
            channel_weights, W_layer, H_layer)

    # Apply the weights on each binary feature map
    expanded_feat_map = np.multiply(expanded_weights, binary_feature_map)

    # Aggregate the feature map of each channel
    feat_map = np.sum(expanded_feat_map, axis=2)

    # Normalize the feature map
    if np.max(feat_map) == 0:
        return feat_map
    feat_map = feat_map / np.max(feat_map)
    return feat_map


def compute_fem(img_array, model, last_conv_layer_name):
    """
    Compute Feature Extraction Maps (FEM) for an input image.

    Parameters:
    img_array (np.ndarray): The input image as a numpy array.
    model (tf.keras.Model): The neural network model.
    last_conv_layer_name (str): The name of the last convolutional layer.

    Returns:
    np.ndarray: Feature Extraction Map (FEM) for the input image.
    """
    model.compile(loss="categorical_crossentropy", optimizer="adam")
    activations = keract.get_activations(model, img_array, auto_compile=True)
    for (k, v) in activations.items():
        if k == last_conv_layer_name:
            feature_map = v
    binary_feature_map = compute_binary_maps(feature_map)
    saliency = aggregate_binary_maps(binary_feature_map, feature_map)
    return saliency
