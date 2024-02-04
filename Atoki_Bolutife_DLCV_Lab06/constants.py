low_res_mask_size = 8
mask_number = 200
threshold = 0.4
size = (224, 224)

# LIME constants
top_labels = 1  # Use top-k labels or not
# RGB color or None (average color of superpixels is used) used to generate neighboring samples
hide_color = [0, 0, 0]
num_lime_features = 100000  # size in number of groups of features of an explanation
num_samples = 2000  # number of perturbated samples to generate
rand_index = 0
# rand_index, hide_color, num_lime_features, num_samples = random_search()
# Rendering parameters
positive_only = True  # display only features having a positive impact on the prediction
negative_only = False  # display only features having a negative impact on the prediction
num_superpixels = 15  # number of superpixels to display
hide_rest = True  # hide the rest of the picture or not
