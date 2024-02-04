# main.py
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from data_loader import load_image, load_fixations, load_ground_truth
from utils import calculate_error_metrics, generate_saliency_map, dict_to_txt, normalise


errors = {}

mae_sum = 0
mse_sum = 0
pcc_sum = 0
ssim_sum = 0

count = 0

def main(images_folder_path, fixations_folder_path, gdfm_folder_path):
    for filename in os.listdir(images_folder_path):
        #Obtaining image, fixation, and GDFM names
        image_name = os.path.splitext(filename)[0]
        fixation_filename = filename.replace("_N_", "_GazeFix_N_").replace(".png", ".txt")
        gdfm_filename = filename.replace("_N_", "_GFDM_N_")
        
        # obtaining paths to image, fixations, and GFDM's
        file_image_path = os.path.join(images_folder_path, filename)
        file_fixation_path = os.path.join(fixations_folder_path, fixation_filename)
        file_gdfm_path = os.path.join(gdfm_folder_path, gdfm_filename)

        #Loading image, fixations and Groundtruth GFDM's
        image = load_image(file_image_path)
        fixation_points = load_fixations(file_fixation_path)
        gdfm_ground_truth = load_ground_truth(file_gdfm_path)

        #Generates Normalized and combined Saliency map for the image using all fixation points
        saliency_map = generate_saliency_map(image, fixation_points)

        #Using colour map to add colour to saliency map
        colormap = plt.get_cmap('jet')  # You can choose a different colormap if desired
        saliency_map_colored = (colormap(saliency_map) * 255).astype(np.uint8)
        
        #Getting pillow images for both grey and coloured saliency maps
        saliency_image_grey = Image.fromarray((saliency_map * 255).astype(np.uint8))
        saliency_image_coloured = Image.fromarray(saliency_map_colored)

        #Blending / Overlaying Fixation map on image
        alpha = 0.5  #parameter for blending strength
        blended_image = Image.blend(image.convert('RGBA'), saliency_image_coloured.convert('RGBA'), alpha)
        
        #Error metrics for comparing Normalized Groundtruth GFDM image with Normalized obtained GFDM image     
        metrics = calculate_error_metrics(normalise(gdfm_ground_truth), normalise(saliency_image_grey))
        errors[image_name] = {
            'MAE': metrics[0],
            'MSE': metrics[1],
            'PCC': metrics[2],
            'SSIM': metrics[3],
        }

        #saving both saliency map and blended image

        os.makedirs('output/GFDM', exist_ok=True)
        os.makedirs('output/blended', exist_ok=True)

        saliency_image_grey.save('output/GFDM/' + filename)
        blended_image.save('output/blended/' + filename)
        # saliency_image_grey.show()


if __name__ == "__main__":
    images_train_folder_path = '/net/ens/DeepLearning/DLCV2023/MexCulture142/images_train'
    images_valid_folder_path = '/net/ens/DeepLearning/DLCV2023/MexCulture142/images_val'
    fixations_folder_path = '/net/ens/DeepLearning/DLCV2023/MexCulture142/fixations'
    gdfm_folder_path = '/net/ens/DeepLearning/DLCV2023/MexCulture142/gazefixationsdensitymaps'

    main(images_train_folder_path, fixations_folder_path, gdfm_folder_path)
    main(images_valid_folder_path, fixations_folder_path, gdfm_folder_path)

    a_MAE = 0
    a_MSE = 0
    a_PCC = 0
    a_SSIM = 0

    dict_to_txt(errors) #adding errors to txt file

    for key, metrics in errors.items():
        mae_sum += metrics['MAE']
        mse_sum += metrics['MSE']
        pcc_sum += metrics['PCC']
        ssim_sum += metrics['SSIM']
        count += 1


    print(f'Average MAE for all images: {mae_sum / count}')
    print(f'Average MSE for all images: {mse_sum / count}')
    print(f'Average PCC for all images: {pcc_sum / count}')
    print(f'Average SSIM for all images: {ssim_sum / count}')

