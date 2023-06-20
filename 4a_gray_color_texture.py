import os
from PIL import Image
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops
from sklearn.preprocessing import StandardScaler

# Define the path to your images directory
images_dir = './watches_images'

# Define the desired image size for preprocessing
image_size = (350, 350)

# Loop through all the subdirectories in the directory
for root, dirs, files in os.walk(images_dir):
    # Loop through all the image files in the subdirectory
    for filename in files:
        # check if the file is an image file
        if filename.endswith('.jpg') or filename.endswith('.png'):
            # Load the image
            img = Image.open(os.path.join(root, filename))
            # Resize the image while maintaining aspect ratio
            img.thumbnail(image_size, Image.ANTIALIAS)
            # Convert the image to RGB mode
            rgb_img = img.convert('RGB')
            # Save the resized image with a new filename
            new_filename = 'processed_' + filename
            rgb_img.save(os.path.join(root, new_filename), 'JPEG')
            # Delete the original image
            os.remove(os.path.join(root, filename))

# Create an empty DataFrame with columns for the image filename, the grayscale histogram features, the color
# histogram features, and the texture features
df_gray = pd.DataFrame(columns=["filename", "gray_features"])
df_color = pd.DataFrame(columns=["filename", "color_features"])
df_texture = pd.DataFrame(columns=["filename", "texture_features"])

# Loop through each subfolder in the directory
for subfolder in os.listdir(images_dir):
    subfolder_path = os.path.join(images_dir, subfolder)
    # Check if the subfolder is actually a directory
    if not os.path.isdir(subfolder_path):
        continue
    # Loop through each image in the subfolder
    for filename in os.listdir(subfolder_path):
        # Ignore non-image files and the .DS_Store file
        if not filename.endswith(('.jpg', '.jpeg', '.png')) or filename == '.DS_Store':
            print(f"Skipping file: {filename}")
            continue
        # Ignore files that don't start with "processed_"
        if not filename.startswith('processed_'):
            print(f"Skipping file: {filename}")
            continue
        # Extract the reference name by removing the "processed_" prefix and file extension
        reference = os.path.splitext(filename[len('processed_'):])[0]
        # Load the image
        img = Image.open(os.path.join(subfolder_path, filename))
        # Convert the image to grayscale
        gray = img.convert('L')
        # Compute the grayscale histogram features
        gray_hist = np.array(gray.histogram())
        gray_hist = gray_hist / np.sum(gray_hist)  # Normalize the histogram so that the values sum to 1
        # Compute the color histogram features
        color_hist = np.array(img.histogram())
        color_hist = color_hist / np.sum(color_hist)  # Normalize the histogram so that the values sum to 1
        # Compute the texture features using graycomatrix and graycoprops
        gray_arr = np.array(gray)
        glcm = graycomatrix(gray_arr, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
        texture_props = np.array([graycoprops(glcm, 'contrast'), graycoprops(glcm, 'energy'), graycoprops(glcm, 'homogeneity'), graycoprops(glcm, 'correlation')]).reshape(-1)
        # Reshape the feature vectors to 1D arrays
        gray_features = gray_hist.reshape(-1)
        color_features = color_hist.reshape(-1)
        # Concatenate the feature vectors
        features = np.concatenate([gray_features]),  # color_features, texture_props])
        # Add the new row to the DataFrame of new data
        df_gray = pd.concat([df_gray, pd.DataFrame({"filename": [reference], "gray_features": [gray_features]})], ignore_index=True)
        df_color = pd.concat([df_color, pd.DataFrame({"filename": [reference], "color_features": [color_features]})],ignore_index=True)
        df_texture = pd.concat([df_texture, pd.DataFrame({"filename": [reference], "texture_features": [texture_props]})], ignore_index=True)

        # Print the filename of the processed image
        print(f"Processed image: {reference}")

# Create a merged dataset which contains all the 3 features
merged_df = pd.merge(df_gray, df_color, on="filename")
merged_df = pd.merge(merged_df, df_texture, on="filename")
merged_df.to_csv("4b_merged_df.csv")