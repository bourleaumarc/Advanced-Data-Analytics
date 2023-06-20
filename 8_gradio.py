import gradio as gr
from PIL import Image
import numpy as np
import pandas as pd
import os
from skimage.feature import graycomatrix, graycoprops
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from torchvision.transforms import transforms
import torch
import torch.nn as nn
from torch.autograd import Variable


def image_analysis(image_array, type_of_suggestions):
    # Create an empty DataFrame with columns for the image filename, the grayscale histogram features, the color
    # histogram features, and the texture features
    df_gray = pd.DataFrame(columns=["filename", "gray_features"])
    df_color = pd.DataFrame(columns=["filename", "color_features"])
    df_texture = pd.DataFrame(columns=["filename", "texture_features"])

    ################################################ Resize and save the image

    # Load the image
    myImage = Image.open(image_array)

    # Convert the image to RGB mode
    rgb_img = myImage.convert('RGB')

    file_path = os.path.join("./watches_images", "image_test.png")
    rgb_img.save(file_path)
    # Convert the image to RGB mode
    rgb_img = rgb_img.convert('RGB')

    ################ checking for device################
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # CNN Network
    class ConvNet(nn.Module):
        def __init__(self, num_classes=20):
            super(ConvNet, self).__init__()

            self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1)
            # Shape= (256,12,350,350)
            self.bn1 = nn.BatchNorm2d(
                num_features=12)  # same number as number of channels; number of different filters or feature maps produced by that layer
            # Shape= (256,12,350,350)
            self.relu1 = nn.ReLU()  # to bring non-linearity
            # Shape= (256,12,350,350)

            self.pool = nn.MaxPool2d(
                kernel_size=2)  # reduces the height and width of convolutional output while keeping the most salient features
            # Reduce the image size be factor 2
            # Shape= (256,12,175,175)

            self.conv2 = nn.Conv2d(in_channels=12, out_channels=20, kernel_size=3, stride=1,
                                   padding=1)  # add second conv layer to apply more patterns and increase the number of channels to 20
            # Shape= (256,20,175,175)
            self.relu2 = nn.ReLU()
            # Shape= (256,20,175,175)

            self.conv3 = nn.Conv2d(in_channels=20, out_channels=32, kernel_size=3, stride=1, padding=1)
            # Shape= (256,32,175,175)
            self.bn3 = nn.BatchNorm2d(num_features=32)
            # Shape= (256,32,175,175)
            self.relu3 = nn.ReLU()
            # Shape= (256,32,175,175)

            self.fc = nn.Linear(in_features=175 * 175 * 32, out_features=num_classes)  # fully connected layer

        # Feed forward function

        def forward(self, input):
            output = self.conv1(input)
            output = self.bn1(output)
            output = self.relu1(output)

            output = self.pool(output)

            output = self.conv2(output)
            output = self.relu2(output)

            output = self.conv3(output)
            output = self.bn3(output)
            output = self.relu3(output)

            # Above output will be in matrix form, with shape (256,32,175,175)

            output = output.view(-1, 32 * 175 * 175)

            output = self.fc(output)

            return output

    checkpoint = torch.load('6b_best_checkpoint.model')
    model=ConvNet(num_classes=20).to(device)
    model.load_state_dict(checkpoint)
    model.eval()  # to set dropout and batch normalisation
    # Transforms
    transformer = transforms.Compose([
        transforms.Resize((350, 350)),
        transforms.ToTensor()])  # 0-255 to 0-1, numpy to tensors

    # Categories
    classes = ['audemars-piguet_images', 'blancpain_images', 'breguet_images', 'breitling_images', 'bulgari_images',
               'cartier_images', 'certina_images', 'chopard_images', 'girard-perregaux_images', 'hublot_images',
               'iwc_images', 'jaeger-lecoultre_images', 'montblanc_images', 'omega_images', 'panerai_images',
               'patek-philippe_images', 'rolex_images', 'tag-heuer_images', 'tissot_images', 'zenith_images']

    # prediction function
    def prediction(img_path, transformer):
        image = Image.open(img_path).convert('RGB')

        image_tensor = transformer(image).float()

        image_tensor = image_tensor.unsqueeze_(0)

        if torch.cuda.is_available():
            image_tensor = image_tensor.cuda()

        input = Variable(image_tensor)

        output = model(input)

        index = output.data.numpy().argmax()  # category id is the one with the highest probability

        pred = classes[index]

        return pred  # output is the category name

    pred_dict = {}
    filename = os.path.basename(file_path)  # Extract the file name from the path
    pred_dict[filename] = prediction(file_path, transformer)
    print(pred_dict)
    brand = pred_dict[filename]

    ################################################ Extract features values
    # Convert the image to grayscale
    gray = rgb_img.convert('L')
    # Compute the grayscale histogram features
    gray_hist = np.array(gray.histogram())
    gray_hist = gray_hist / np.sum(gray_hist)  # Normalize the histogram so that the values sum to 1
    # Compute the color histogram features
    color_hist = np.array(rgb_img.histogram())
    color_hist = color_hist / np.sum(color_hist)  # Normalize the histogram so that the values sum to 1
    # Compute the texture features using graycomatrix and graycoprops
    gray_arr = np.array(gray)
    glcm = graycomatrix(gray_arr, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    texture_props = np.array(
        [graycoprops(glcm, 'contrast'), graycoprops(glcm, 'energy'), graycoprops(glcm, 'homogeneity'),
         graycoprops(glcm, 'correlation')]).reshape(-1)
    # Reshape the feature vectors to 1D arrays
    gray_features = gray_hist.reshape(-1)
    color_features = color_hist.reshape(-1)

    df_gray = pd.concat([df_gray, pd.DataFrame({"filename": [file_path], "gray_features": [gray_features]})],
                        ignore_index=True)
    df_color = pd.concat([df_color, pd.DataFrame({"filename": [file_path], "color_features": [color_features]})],
                         ignore_index=True)
    df_texture = pd.concat([df_texture, pd.DataFrame({"filename": [file_path], "texture_features": [texture_props]})],
                           ignore_index=True)
    merged_df = pd.merge(df_gray, df_color, on="filename")
    merged_df = pd.merge(merged_df, df_texture, on="filename")

    # Load the existing dataset
    # Rename the variable "image file" to "Reference" in 9d_merged_df_copy
    data_change = pd.read_csv("4b_merged_df.csv")
    data2 = pd.read_csv("3b_data_with_images.csv")
    data2.rename(columns={"Reference": "filename"}, inplace=True)

    # Merge the "Brand" variable from 5_data_with_images into 9d_merged_df_copy based on "filename"
    merge = data_change.merge(data2[["filename", "Brand"]], on="filename", how="left")

    # Load the existing dataset
    merge.to_csv('8b_merged.csv')
    existing_dataset = pd.read_csv('8b_merged.csv')

    # Concatenate the existing dataset and the newly created DataFrame
    new_dataset = pd.concat([existing_dataset, merged_df], ignore_index=True)

    # Save the new dataset to a CSV file
    new_dataset.to_csv('8c_final_merge.csv', index=False)
    df = pd.read_csv('8c_final_merge.csv')

    ################################################ Standardization of features values

    # Extract the features into separate arrays
    gray_features = df['gray_features'].apply(lambda x: np.fromstring(x[1:-1], sep=' ')).values
    color_features = df['color_features'].apply(lambda x: np.fromstring(x[1:-1], sep=' ')).values
    texture_features = df['texture_features'].apply(lambda x: np.fromstring(x[1:-1], sep=' ')).values

    # Create a StandardScaler object
    scaler = StandardScaler()

    # Scale the individual feature arrays
    gray_features_scaled = np.vstack(gray_features).astype(float)
    gray_features_scaled = scaler.fit_transform(gray_features_scaled).tolist()

    color_features_scaled = np.vstack(color_features).astype(float)
    color_features_scaled = scaler.fit_transform(color_features_scaled).tolist()

    texture_features_scaled = np.vstack(texture_features).astype(float)
    texture_features_scaled = scaler.fit_transform(texture_features_scaled).tolist()

    # Update the data DataFrame with the scaled features
    df['gray_features'] = gray_features_scaled
    df['color_features'] = color_features_scaled
    df['texture_features'] = texture_features_scaled

    df.to_csv('8d_temp_df.csv')

    ################################################ KNN

    # Read the CSV file into a pandas DataFrame
    data = pd.read_csv('8d_temp_df.csv')
    # Change the Brand names to be equal to the name saved under the classes vector
    # Mapping of brand names
    brand_mapping = {
        'Audemars Piguet': 'audemars-piguet_images',
        'Blancpain': 'blancpain_images',
        'Breguet': 'breguet_images',
        'Breitling': 'breitling_images',
        'Bulgari': 'bulgari_images',
        'Cartier': 'cartier_images',
        'Certina': 'certina_images',
        'Chopard': 'chopard_images',
        'Girard-Perregaux': 'girard-perregaux_images',
        'Hublot': 'hublot_images',
        'IWC': 'iwc_images',
        'Jaegger-LeCoultre': 'jaeger-lecoultre_images',
        'Montblanc': 'montblanc_images',
        'Omega': 'omega_images',
        'Panerai': 'panerai_images',
        'Patek Philippe': 'patek-philippe_images',
        'Rolex': 'rolex_images',
        'TAG Heuer': 'tag-heuer_images',
        'Tissot': 'tissot_images',
        'Zenith': 'zenith_images'
    }

    # Apply brand mapping and update the 'brand' column
    data['Brand'] = data['Brand'].map(brand_mapping)

    if type_of_suggestions == "Same brand":
        # Filter the data based on the brand value
        filtered_data = data[data['Brand'] == brand]
        # Save the filtered data to a CSV file
        filtered_data.to_csv('8e_filtered_data.csv', index=False)

        # Extract the gray features and reference names into separate arrays
        gray_features = filtered_data['gray_features'].apply(lambda x: np.fromstring(x[1:-1], sep=' '))
        color_features = filtered_data['color_features'].apply(lambda x: np.fromstring(x[1:-1], sep=' '))
        texture_features = filtered_data['texture_features'].apply(lambda x: np.fromstring(x[1:-1], sep=' '))

        reference_names = filtered_data['filename'].values

    else:
        # Extract the gray features and reference names into separate arrays
        gray_features = data['gray_features'].apply(lambda x: np.fromstring(x[1:-1], sep=' '))
        color_features = data['color_features'].apply(lambda x: np.fromstring(x[1:-1], sep=' '))
        texture_features = data['texture_features'].apply(lambda x: np.fromstring(x[1:-1], sep=' '))

        reference_names = data['filename'].values

    # Reshape the gray features to a 2D array
    X_gray = np.vstack(gray_features)
    X_color = np.vstack(color_features)
    X_texture = np.vstack(texture_features)

    # Concatenate the variables together into a matrix
    X = np.concatenate((X_gray, X_color, X_texture), axis=1)

    # Create the target vector y with the reference names
    y = reference_names

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the KNN classifier
    knn = KNeighborsClassifier(n_neighbors=2)

    # Train the KNN model using only the gray features
    knn.fit(X_train, y_train)

    # Predict the closest neighbors for a specific watch (example)
    watch_features = X[-1]  # Replace with the features of the watch you want to find neighbors for

    closest_neighbors = knn.kneighbors([watch_features], n_neighbors=3)

    # Retrieve the indices of the closest neighbors
    neighbor_indices = closest_neighbors[1][0]

    # Retrieve the unique identifiers of the closest neighbors
    closest_watch_ids = [reference_names[idx] for idx in neighbor_indices]

    # Get the corresponding image paths for the closest watch IDs
    image_folder = "watches_images_no_category"
    image_paths = [os.path.join(image_folder, f"{watch_id}.jpg") for watch_id in closest_watch_ids[:3]]

    # Load the images and resize them
    watch_images = [Image.open(image_path).resize((450, 450), Image.ANTIALIAS) for image_path in image_paths]

    return closest_watch_ids, watch_images[0], watch_images[1], watch_images[2]



################################################ Formatting Gradio UI

title = "Watches Recommendations"

css = """
    .output-img {
        width: 20px;
        height: 20px;
    }
"""

demo = gr.Interface(
    fn=image_analysis,
    inputs=[
        gr.inputs.Textbox(label="Input the absolute path of the image"),
        gr.inputs.Dropdown(["Any brand", "Same brand"], label="Type of suggestions", type="value")
    ],
    outputs=[
        gr.outputs.Textbox(label="Closest Watch IDs"),
        gr.outputs.Image(label="Watch Image 1", type="pil"),
        gr.outputs.Image(label="Watch Image 2", type="pil"),
        gr.outputs.Image(label="Watch Image 3", type="pil")
    ],
    css=css,
    title=title
)

demo.launch()