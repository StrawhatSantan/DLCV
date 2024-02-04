import numpy as np
import tensorflow as tf
import functools
import keract

from utils import get_last_layer_name

# Display
# from IPython.display import Image, display
import matplotlib.pyplot as plt
import matplotlib.cm as cm


class FEM:
    """
    A class for implementing FEM (Feature Explanation method) for classification explainability.
    This class provides methods to compute and visualize activation maps
    highlighting the regions that are most important for a given class prediction.

    Attributes:
        model_name (str): The name of the deep learning model used for prediction.
        img_array (numpy.ndarray): The input image as a NumPy array.
        weight_activation_maps (numpy.ndarray): The weighted activation maps.
        last_conv_layer_output (numpy.ndarray): The output of the last convolutional layer.
        saliency_map (numpy.ndarray): The computed saliency map.
    """

    def __init__(self, model, model_name, img_array):
        """
        Initialize a GradCAM instance.

        Args:
            model_name (str): The name of the deep learning model.
            img_array (numpy.ndarray): The input image as a NumPy array.
        """
        self.model = model
        self.model_name = model_name
        self.img_array = img_array

    def expand_flat_values_to_activation_shape(self, values, W_layer, H_layer):
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

    def compute_binary_maps(self, feature_map, sigma=None):
        batch_size, W_layer, H_layer, N_channels = feature_map.shape
        thresholded_tensor = np.zeros(
            (batch_size, W_layer, H_layer, N_channels))

        if sigma is None:
            feature_sigma = 2
        else:
            feature_sigma = sigma

        for B in range(batch_size):
            # Get the activation value of the current sample
            activation = feature_map[B, :, :, :]

            # Calculate its mean and its std per channel
            mean_activation_per_channel = tf.reduce_mean(
                activation, axis=[0, 1])

            std_activation_per_channel = tf.math.reduce_std(
                activation, axis=(0, 1))

            assert len(mean_activation_per_channel) == N_channels
            assert len(std_activation_per_channel) == N_channels

            # Transform the mean in the same shape than the activation maps
            mean_activation_expanded = tf.reshape(mean_activation_per_channel, (
                1, 1, -1)) * np.ones((W_layer, H_layer, len(mean_activation_per_channel)))

            # Transform the std in the same shape than the activation maps
            std_activation_expanded = tf.reshape(std_activation_per_channel, (
                1, 1, -1)) * np.ones((W_layer, H_layer, len(std_activation_per_channel)))

            # Build the binary map
            thresholded_tensor[B, :, :, :] = tf.cast((activation > (
                mean_activation_expanded + feature_sigma * std_activation_expanded)), dtype=tf.int32)

        return thresholded_tensor

    def aggregate_binary_maps(self, binary_feature_map, orginal_feature_map):
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
            expanded_weights = self.expand_flat_values_to_activation_shape(
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

    def compute_saliency_map(self,):
        """
        Compute Feature Extraction Maps (FEM) for an input image.

        Parameters:
        img_array (np.ndarray): The input image as a numpy array.
        model (tf.keras.Model): The neural network model.
        last_conv_layer_name (str): The name of the last convolutional layer.

        Returns:
        np.ndarray: Feature Extraction Map (FEM) for the input image.

        """
        last_conv_layer_name = get_last_layer_name(self.model_name)
        self.model.layers[-1].activation = None
        # self.model.compile(loss="categorical_crossentropy", optimizer="adam")

        fem_model = tf.keras.models.Model(inputs=self.model.input,
                                          outputs=self.model.get_layer(last_conv_layer_name).output)

        feature_map = fem_model(self.img_array)

        binary_feature_map = self.compute_binary_maps(feature_map)

        saliency_map = self.aggregate_binary_maps(
            binary_feature_map, feature_map)

        return saliency_map
