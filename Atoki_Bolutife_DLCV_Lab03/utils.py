import numpy as np
import os
import random
import matplotlib.pyplot as plt


def normalise(matrix):
    max_value = np.max(matrix)
    return matrix / max_value


def grid_layout(images, titles):
    """"
    Inputs:
    List of images to be plotted
    List of titles for these images

    Output:
    Displays a 2x3 grid of the images and corresponding title and saves grid image
    """
    # Creates a 2x3 grid for the images
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))

    # Sets the title for the entire grid
    fig.suptitle('Grid layout of Results', fontsize=16)

    # Loops through images and titles, and plot them in the grid
    for i, (image, title) in enumerate(zip(images, titles)):
        row = i // 3
        col = i % 3
        ax = axes[row, col]
        ax.imshow(image)
        ax.set_title(title)
        ax.axis('off')

    # Adjusts the layout
    plt.tight_layout(pad=2.0)

    os.makedirs('output_images', exist_ok=True)
    save_index = random.randint(1, 100)
    plt.savefig('output_images/' + str(save_index) + '.jpg')

    # Shows the grid
    plt.show()
