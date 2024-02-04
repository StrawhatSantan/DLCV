# Import necessary libraries and modules
import numpy as np
import tensorflow as tf
import cv2
from skimage.transform import resize
from tensorflow.keras.models import Model
from utils import normalise

# Import custom utility functions from 'utils' module
from utils import get_last_layer_name


class GradCAM:
    """
    A class for implementing GradCAM (Gradient-weighted Class Activation Mapping).
    This class provides methods to compute and visualize activation maps
    highlighting the regions that are most important for a given class prediction.

    Attributes:
        model_name (str): The name of the deep learning model used for prediction.
        img_array (numpy.ndarray): The input image as a NumPy array.
        weight_activation_maps (numpy.ndarray): The weighted activation maps.
        last_conv_layer_output (numpy.ndarray): The output of the last convolutional layer.
        saliency_map (numpy.ndarray): The computed saliency map.
    """

    def __init__(self, model, model_name, img_array, class_index):
        """
        Initialize a GradCAM instance.

        Args:
            model_name (str): The name of the deep learning model.
            img_array (numpy.ndarray): The input image as a NumPy array.
        """
        self.model = model
        self.model_name = model_name
        self.img_array = img_array
        self.class_index = class_index

        self.weight_activation_maps = None
        self.last_conv_layer_output = None

    def get_model(self):
        """
        Create and return the GradCAM model.

        Returns:
            tensorflow.keras.models.Model: The GradCAM model.
        """

        # Removing the softmax activation from the last layer
        self.model.layers[-1].activation = None

        # Get the name of the last convolutional layer
        last_layer_name = get_last_layer_name(self.model_name)
        last_conv_layer = self.model.get_layer(last_layer_name)

        grad_cam_model = Model(self.model.inputs, [
                               self.model.output, last_conv_layer.output])

        return grad_cam_model

    def compute_gradients(self, grad_cam_model, class_index):
        """
        Compute gradients of the class prediction with respect to the last convolutional layer.

        Args:
            grad_cam_model (tensorflow.keras.models.Model): The GradCAM model.
            class_name (str): The target class name for which to compute gradients.

        Returns:
            tensorflow.Tensor: The computed gradients.
        """
        if self.last_conv_layer_output is not None:
            return self.last_conv_layer_output

        # preprocess_input = self.get_preprocess_input()
        # img_array = tf.expand_dims(preprocess_input(self.img_array), axis=0)

        # Use GradientTape to compute gradients
        with tf.GradientTape() as tape:
            preds, last_conv_layer_output = grad_cam_model(self.img_array)

            # Getting the score for the class indexx
            score = preds[0][class_index]

        # Computing gradients
        gradients = tape.gradient(score, last_conv_layer_output)

        # Storing the last convolutional layer's output
        self.last_conv_layer_output = last_conv_layer_output

        return gradients

    def pool_gradients(self, gradients):
        """
        Pool gradients by performing global average pooling for each channel.

        Args:
            gradients (tensorflow.Tensor): The computed gradients.

        Returns:
            List[tensorflow.Tensor]: The pooled gradients for each channel.
        """
        if self.weight_activation_maps is not None:
            return self.weight_activation_maps

        pooled_gradients = []
        for channel_index in range(gradients.shape[-1]):
            pooled_value = tf.keras.layers.GlobalAveragePooling2D()(
                tf.expand_dims(gradients[:, :, :, channel_index], axis=-1))
            pooled_gradients.append(pooled_value)

        return pooled_gradients

    def weight_activation_map(self, pooled_gradients):
        """
        Compute weighted activation maps by multiplying gradients and the last layer's output.

        Args:
            pooled_gradients (List[tensorflow.Tensor]): The pooled gradients for each channel.
        """
        if self.weight_activation_maps is not None:
            return self.weight_activation_maps

        shape = self.last_conv_layer_output.shape.as_list()[1:]
        weighted_maps = np.empty(shape)

        if self.last_conv_layer_output.shape[-1] != len(pooled_gradients):
            print('error, size mismatch')
        else:
            for i in range(len(pooled_gradients)):
                weighted_maps[:, :, i] = np.squeeze(self.last_conv_layer_output.numpy()
                                                    [:, :, :, i], axis=0) * pooled_gradients[i]

        # Store the weighted activation maps
        self.weight_activation_maps = weighted_maps

    def apply_relu(self):
        """
        Apply the ReLU activation function to the weighted activation maps.
        Sets negative values to 0.
        """
        if self.weight_activation_maps is not None:
            self.weight_activation_maps[self.weight_activation_maps < 0] = 0

    def apply_dimension_average_pooling(self):
        """
        Apply global average pooling along the channel dimension.

        Returns:
            numpy.ndarray: The result of global average pooling.
        """
        if self.weight_activation_maps is not None:
            return np.mean(self.weight_activation_maps, axis=2)

    def compute_saliency_map(self):
        """
        Compute the saliency map using the GradCAM approach.
        """

        # Obtaining GRADCAM model
        grad_cam_model = self.get_model()

        # obtaining gradients of last conv layer wrt to prediction of class of input image
        gradients = self.compute_gradients(grad_cam_model, self.class_index)

        # Performing global average pooling of gradients to get 1x1 scalar
        pooled_gradients = self.pool_gradients(gradients)

        # weighting each activation map by its corresponding scalar value
        self.weight_activation_map(pooled_gradients)

        # Setting negative values in weighted maps to 0
        self.apply_relu()

        # Performing dimension pooling to get 1 channel array to get saliency map
        saliency_map = self.apply_dimension_average_pooling()

        # normalizing saliency map
        saliency_map = normalise(saliency_map)

        # smoothening heatmap
        # Define the standard deviation (sigma) for the Gaussian blur
        sigma = 0.5

        # Apply Gaussian blur to the heatmap
        self.saliency_map = cv2.GaussianBlur(saliency_map, (0, 0), sigma)

        return saliency_map
