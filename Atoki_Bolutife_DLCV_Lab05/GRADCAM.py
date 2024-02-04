# Import necessary libraries and modules
import numpy as np
import tensorflow as tf
from skimage.transform import resize
from tensorflow.keras.models import Model

# Import custom utility functions from 'utils' module
from utils import make_classifier, get_last_layer_name


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
    """

    def __init__(self, model_name, img_array):
        """
        Initialize a GradCAM instance.

        Args:
            model_name (str): The name of the deep learning model.
            img_array (numpy.ndarray): The input image as a NumPy array.
        """
        self.model_name = model_name
        self.img_array = self.resize_array(img_array)
        self.weight_activation_maps = 0
        self.last_conv_layer_output = 0

    def get_decode_predictions(self):
        """
        Get the appropriate decode_predictions function based on the model_name.

        Returns:
            function: The decode_predictions function for the model.
        """
        if self.model_name == 'Xception':
            decode_predictions = tf.keras.applications.xception.decode_predictions
        elif self.model_name == 'ResNet':
            decode_predictions = tf.keras.applications.resnet_v2.decode_predictions
        return decode_predictions

    def get_preprocess_input(self):
        """
        Get the appropriate preprocess_input function based on the model_name.

        Returns:
            function: The preprocess_input function for the model.
        """
        if self.model_name == 'Xception':
            preprocess_input = tf.keras.applications.xception.preprocess_input
        elif self.model_name == 'ResNet':
            preprocess_input = tf.keras.applications.resnet_v2.preprocess_input
        return preprocess_input

    def get_model(self):
        """
        Create and return the GradCAM model.

        Returns:
            tensorflow.keras.models.Model: The GradCAM model.
        """
        # Create a classification model
        model = make_classifier(self.model_name)

        # Removing the softmax activation from the last layer
        model.layers[-1].activation = None

        # Get the name of the last convolutional layer
        last_layer_name = get_last_layer_name(self.model_name)

        # Create the GradCAM model, which takes an input image and returns the
        # output of the last convolutional layer and the model's predictions
        grad_cam_model = Model(
            model.inputs, [model.get_layer(
                last_layer_name).output, model.output]
        )

        return grad_cam_model

    def resize_array(self, img_array):
        """
        Resize the input image array based on the model's input size.

        Args:
            img_array (numpy.ndarray): The input image as a NumPy array.

        Returns:
            numpy.ndarray: The resized image array.
        """
        if self.model_name == 'Xception':
            size = (299, 299)
        elif self.model_name == 'ResNet':
            size = (224, 224)
        return resize(img_array, size)

    def compute_gradients(self, grad_cam_model, class_name):
        """
        Compute gradients of the class prediction with respect to the last convolutional layer.

        Args:
            grad_cam_model (tensorflow.keras.models.Model): The GradCAM model.
            class_name (str): The target class name for which to compute gradients.

        Returns:
            tensorflow.Tensor: The computed gradients.
        """
        preprocess_input = self.get_preprocess_input()
        img_array = tf.expand_dims(preprocess_input(self.img_array), axis=0)

        # Use GradientTape to compute gradients
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_cam_model(img_array)
            sorted_preds = tf.sort(preds, direction='DESCENDING')

            decode_predictions = self.get_decode_predictions()

            labels = decode_predictions(np.asarray(preds), top=1000)[0]

            # Find the index of the target class
            class_index = [index for index, x in enumerate(
                labels) if x[1] == class_name][0]

            # Calculate the score for the target class
            score = sorted_preds[:, class_index]

        # Compute gradients
        gradients = tape.gradient(score, last_conv_layer_output)

        # Store the last convolutional layer's output
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
        self.weight_activation_maps[self.weight_activation_maps < 0] = 0

    def apply_dimension_average_pooling(self):
        """
        Apply global average pooling along the channel dimension.

        Returns:
            numpy.ndarray: The result of global average pooling.
        """
        return np.mean(self.weight_activation_maps, axis=2)

# import numpy as np
# import tensorflow as tf
# from skimage.transform import resize
# from tensorflow.keras.models import Model

# from utils import (make_classifier, get_last_layer_name)


# class GradCAM:
#     def __init__(self, model_name, img_array):
#         self.model_name = model_name
#         self.img_array = self.resize_array(img_array)
#         self.weight_activation_maps = 0
#         self.last_conv_layer_output = 0

#     def get_decode_predcitions(self):
#         if self.model_name == 'Xception':
#             decode_predictions = tf.keras.applications.xception.decode_predictions

#         elif self.model_name == 'ResNet':
#             decode_predictions = tf.keras.applications.resnet_v2.decode_predictions
#         return decode_predictions

#     def get_preprocess_input(self):
#         if self.model_name == 'Xception':
#             preprocess_input = tf.keras.applications.xception.preprocess_input

#         elif self.model_name == 'ResNet':
#             preprocess_input = tf.keras.applications.resnet_v2.preprocess_input
#         return preprocess_input

#     def get_model(self):
#         model = make_classifier(self.model_name)

#         # Removing last layer's softmax
#         model.layers[-1].activation = None

#         # Getting last layer's name
#         last_layer_name = get_last_layer_name(self.model_name)

#         grad_cam_model = Model(
#             model.inputs, [model.get_layer(
#                 last_layer_name).output, model.output]
#         )

#         return grad_cam_model

#     def resize_array(self, img_array):
#         if self.model_name == 'Xception':
#             size = (299, 299)

#         elif self.model_name == 'ResNet':
#             size = (224, 224)

#         return resize(img_array, size)

#     def compute_gradients(self, grad_cam_model, class_name):

#         preprocess_input = self.get_preprocess_input()
#         img_array = tf.expand_dims(preprocess_input(self.img_array), axis=0)

#         # Computing the gradient of the class prediction with respect to the activations of the last conv layer
#         with tf.GradientTape() as tape:
#             last_conv_layer_output, preds = grad_cam_model(img_array)
#             sorted_preds = tf.sort(preds, direction='DESCENDING')

#             decode_predictions = self.get_decode_predcitions()

#             labels = decode_predictions(np.asarray(preds), top=1000)[0]

#             class_index = [index for index, x in enumerate(
#                 labels) if x[1] == class_name][0]

#             score = sorted_preds[:, class_index]

#         gradients = tape.gradient(score, last_conv_layer_output)

#         self.last_conv_layer_output = last_conv_layer_output

#         return gradients

#     def pool_gradients(self, gradients):
#         pooled_gradients = []
#         for channel_index in range(gradients.shape[-1]):
#             pooled_value = tf.keras.layers.GlobalAveragePooling2D()(
#                 tf.expand_dims(gradients[:, :, :, channel_index], axis=-1))
#             pooled_gradients.append(pooled_value)

#         return pooled_gradients

#     def weight_activation_map(self, pooled_gradients):
#         shape = self.last_conv_layer_output.shape.as_list()[1:]
#         weighted_maps = np.empty(shape)

#         if self.last_conv_layer_output.shape[-1] != len(pooled_gradients):
#             print('error, size mismatch')
#         else:
#             for i in range(len(pooled_gradients)):
#                 weighted_maps[:, :, i] = np.squeeze(self.last_conv_layer_output.numpy()
#                                                     [:, :, :, i], axis=0) * pooled_gradients[i]

#         self.weight_activation_maps = weighted_maps

#     def apply_relu(self):
#         # setting negative values to 0
#         self.weight_activation_maps[self.weight_activation_maps < 0] = 0

#     def apply_dimension_average_pooling(self):
#         return np.mean(
#             self.weight_activation_maps, axis=2)
