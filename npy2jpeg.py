import os
import numpy as np
from PIL import Image
import argparse

def save_images_from_numpy(array, output_dir, prefix='image'):
    """
    Save images from a NumPy array to a specified directory as JPEG files.
    
    Parameters:
        array (numpy.ndarray): Input array of shape (N, H, W, C), where
                               N is the number of images,
                               H is the height,
                               W is the width,
                               C is the number of color channels (3 for RGB, 1 for grayscale).
        output_dir (str): Path to the directory where images will be saved.
        prefix (str): Prefix for output image filenames.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for i, img_array in enumerate(array):
        img = Image.fromarray(np.uint8(img_array))  # Convert to 8-bit image
        img_path = os.path.join(output_dir, f"{prefix}_{i}.jpg")
        img.save(img_path, "JPEG")
    
    print(f"Saved {len(array)} images to {output_dir}")

# Argument parser
ap = argparse.ArgumentParser()
ap.add_argument('-n', '--npy', type=str, default="calib_pose_detection_224_dataset", help="Numpy array to convert. Default is pose detect dataset (calib_pose_detection_224_dataset.npy)")
args = ap.parse_args()

if args.npy != "":
    npy_string = args.npy
    array_path = f"{npy_string}.npy"
    output_dir = f"{npy_string}_jpeg"
    if os.path.exists(array_path):
        dataset = np.load(array_path)
        save_images_from_numpy(dataset, output_dir)
    else:
        print(f"Error: {array_path} not found.")
else:
    print("Error: No numpy array specified. Use -n or --npy to specify the numpy array.")
