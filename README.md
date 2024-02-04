# Deep Learning for Computer Vision Labs

This repository contains lab submissions for a Deep Learning for Computer Vision course. The labs involve various tasks related to gaze fixation density maps, explanation methods, and model evaluation. Below is an overview of the labs conducted:

## Lab 1: Finetuning ResNet-50-V2 on MexCulture Dataset

- Loading the daatset, model architecture amd pre-trained weights
- Performing finetuning for classfication and saving model weights
- Displaying wrongly classified labels

## Lab 2: Gaze Fixation Density Maps (GFDMaps)

- Creation of Gaze Fixation Density maps from gaze points.
- Evaluation of Gaze Fixation Density maps using specific metrics.

## Lab 3: Representation Methods

- Representation of Gaze Fixation Density maps by isolines and contours.
- Representation of Gaze Fixation Density maps by soft and hard representations.

## Lab 4: Black Box Explanation Methods

Implementation of two black-box explanation methods for pretrained ResNet models:

- **RISE (Randomized Input Sampling Explanation)**
- **LIME (Local Interpretable Model-agnostic Explanations)**

## Lab 5: White Box Explanation Methods

Implementation of two white-box explanation methods for pretrained ResNet models:

- **GRADCAM (Gradient-weighted Class Activation Mapping)**
- **FEM (Feature Extraction Maps)**

## Lab 6: Evaluation of Explanation Maps

Loading finetuned model from lab one and performing all 4 explanation methods on it, and then performing Assessment of obtained explanation maps using the following evaluation metrics:

- **Insertion**
- **Deletion**
- **Pearson Correlation Coefficient (PCC)**
- **Structural Similarity Index (SSIM)**

## Structure of the Repository

- Each lab is organized in its respective directory.
- Detailed instructions, code, and documentation for each lab are available within their corresponding directories.

## Instructions

Navigate to each lab directory for detailed instructions, code, and documentation, and report.

Feel free to explore, learn, and contribute to the development of advanced techniques in computer vision and deep learning. If you have any questions or suggestions, please don't hesitate to reach out.

Happy coding!
