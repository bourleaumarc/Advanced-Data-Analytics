import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Read the CSV file into a pandas DataFrame
data = pd.read_csv('4b_merged_df.csv')

# Extract the features into separate arrays
gray_features = data['gray_features'].apply(lambda x: np.fromstring(x[1:-1], sep=' ')).values
color_features = data['color_features'].apply(lambda x: np.fromstring(x[1:-1], sep=' ')).values
texture_features = data['texture_features'].apply(lambda x: np.fromstring(x[1:-1], sep=' ')).values

print("Shapes:")
print("gray_features shape:", gray_features.shape)
print("color_features shape:", color_features.shape)
print("texture_features shape:", texture_features.shape)

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
data['gray_features'] = gray_features_scaled
data['color_features'] = color_features_scaled
data['texture_features'] = texture_features_scaled

# Save the updated DataFrame to a CSV file
data.to_csv("5b_merged_df_scaled.csv", index=False)

