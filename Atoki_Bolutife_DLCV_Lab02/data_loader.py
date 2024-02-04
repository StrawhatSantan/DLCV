# data_loader.py
import os
from PIL import Image

def load_image(file_path):
    return Image.open(file_path)

def load_fixations(file_path):
    fixations = []
    with open(file_path, 'r') as file:
        for line in file:
            columns = line.strip().split()
            fixation_point = (int(columns[0]), int(columns[1]))
            fixations.append(fixation_point)
    return fixations

def load_ground_truth(file_path):
    return Image.open(file_path)
