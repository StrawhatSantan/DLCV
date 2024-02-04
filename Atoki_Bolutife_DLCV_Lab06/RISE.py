# Import necessary libraries and modules
import tensorflow as tf
from skimage.transform import resize
import numpy as np
from constants import size


class RISE:
    def __init__(self, model, img_array, class_index, n_masks, mask_size, threshold):

        self.model = model
        self.img_array = img_array
        self.class_index = class_index
        self.n_masks = n_masks
        self.mask_size = mask_size
        self.threshold = threshold

        self.perturbed_images = None
        self.masks = None
        self.scores = None

    def generate_masks(self):
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
        if self.perturbed_images is not None and self.masks is not None:
            return self.perturbed_images, self.masks

        H, W = size

        # using threshold for deciding mask value; mask_value >= threshold --> 1
        masks = np.empty((self.n_masks, H, W))
        perturbed_images = np.empty((self.n_masks, H, W, 3))

        # Generate masks
        for i in range(self.n_masks):
            grid = (np.random.rand(1, self.mask_size, self.mask_size)
                    < self.threshold).astype("float32")

            # Mask generation algorithm
            C_H, C_W = np.ceil(H / self.mask_size), np.ceil(W / self.mask_size)
            h_new_mask, w_new_mask = (
                self.mask_size + 1) * C_H, (self.mask_size + 1) * C_W

            x, y = np.random.randint(0, C_H), np.random.randint(0, C_W)

            masks[i, :, :] = resize(
                grid[0],
                (h_new_mask, w_new_mask),
                order=1,
                mode="reflect",
                anti_aliasing=False,
            )[x: x + H, y: y + W]

            # Obtaining mask in 3 channels
            mask_3d = masks[i, :, :][..., None].repeat(3, axis=2)

            # blending mask and image
            pertubed_image = mask_3d * self.img_array

            perturbed_images[i, :, :, :] = pertubed_image

        # # Store the generated perturbed images and masks
        self.perturbed_images, self.masks = perturbed_images, masks

        # return perturbed_images, masks

    def obtain_prediction_scores(self):
        """
        Make predictions using a pre-trained deep learning model.

        Returns:
            list: List of scores.
        """
        if self.scores is not None:
            return self.scores

        scores = []

        for image_index in range(self.perturbed_images.shape[0]):
            image = self.perturbed_images[image_index, :, :, :]

            # Expanding dimension
            image = tf.expand_dims(image, axis=0)

            # Performing prediction
            predictions = self.model.predict(image).flatten()

            # Getting the score for the class index
            score = predictions[self.class_index]
            print(score)

            scores.append(score)

        # Store the computed scores
        self.scores = scores

        # return scores

    def weight_saliency_maps(self):
        sum_of_scores = np.sum(self.scores)
        saliency_map = np.zeros(self.masks[0].shape, dtype=np.float64)

        for i, mask_i in enumerate(self.masks):
            score_i = self.scores[i]
            saliency_map += score_i * mask_i

        saliency_map /= sum_of_scores

        # Store the calculated saliency map
        # self.saliency_map = saliency_map

        return saliency_map

    def compute_saliency_map(self):
        """
        Calculate a saliency map based on scores and masks.

        Returns:
            numpy.ndarray: The calculated saliency map.
        """

        # Obtain generated masks and perturbed images
        self.generate_masks()

        # Obtain scores for perturbed images
        self.obtain_prediction_scores()

        # Weights and aggregates saliency maps according to prediction scores
        saliency_map = self.weight_saliency_maps()

        # return np.log(saliency_map)
        return saliency_map
