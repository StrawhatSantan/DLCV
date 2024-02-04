# Generate saliency with LIME algorithm
from skimage.segmentation import mark_boundaries
from lime import lime_image
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


def get_lime_explanation(model, img_array, pred_index, top_labels, hide_color, num_lime_features, num_samples):
    explainer = lime_image.LimeImageExplainer(
        random_state=0)  # for reproductibility

    img_array = img_array.numpy().astype(np.float64)
    explanation = explainer.explain_instance(
        img_array,
        model.predict,
        top_labels=top_labels,
        labels=(pred_index,),
        hide_color=hide_color,
        num_features=num_lime_features,
        num_samples=num_samples,
        random_seed=0)  # for reproductibility

    return explanation


def explain_with_lime(model, img_array,
                      top_labels, hide_color, num_lime_features, num_samples,  # Explanation parameters
                      positive_only, negative_only, num_superpixels, hide_rest,  # Rendering parameters
                      rand_index):  # hidden colour parameters

    hidden_colour = ''
    if rand_index == 0:
        hidden_colour = 'No'
    elif rand_index == 1:
        hidden_colour = 'Red'
    elif rand_index == 2:
        hidden_colour = 'Green'
    elif rand_index == 3:
        hidden_colour = 'Blue'

    preds = model.predict(img_array).flatten()
    pred_index = np.argmax(preds)

    explanation = get_lime_explanation(model, img_array[0],
                                       pred_index, top_labels, hide_color, num_lime_features, num_samples)

    index = explanation.top_labels[0]
    dict_heatmap = dict(explanation.local_exp[index])
    heatmap = np.vectorize(dict_heatmap.get)(explanation.segments)

    heatmap = np.array(heatmap)

    temp, mask = explanation.get_image_and_mask(label=pred_index,
                                                positive_only=positive_only, negative_only=negative_only, num_features=num_superpixels, hide_rest=hide_rest)

    return heatmap
