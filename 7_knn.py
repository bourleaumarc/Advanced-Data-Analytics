import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Read the CSV file into a pandas DataFrame
data = pd.read_csv('5b_merged_df_scaled.csv')

# Extract the gray, color, and texture features into separate arrays
gray_features = data['gray_features'].apply(lambda x: np.fromstring(x[1:-1], sep=' '))
color_features = data['color_features'].apply(lambda x: np.fromstring(x[1:-1], sep=' '))
texture_features = data['texture_features'].apply(lambda x: np.fromstring(x[1:-1], sep=' '))

reference_names = data['filename'].values

# Concatenate the features together into a matrix
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

# Train the KNN model using all features
knn.fit(X_train, y_train)

# Define the indexes of the reference names for which you want to find neighbors
# Example: finding the neighbors of watches at index 0, 1 and 2
indexes_to_find_neighbors = [0, 1, 2]

# Iterate over the specified indexes
for i in indexes_to_find_neighbors:
    # Get the features for the current reference name
    watch_features = X[i]
    print("Input watch:", reference_names[i])

    # Find the closest neighbors for the current watch
    closest_neighbors = knn.kneighbors([watch_features], n_neighbors=3)

    # Retrieve the indices of the closest neighbors
    neighbor_indices = closest_neighbors[1][0]

    # Retrieve the reference names of the closest neighbors
    closest_watch_ids = [reference_names[idx] for idx in neighbor_indices]

    print("Closest neighbors:", closest_watch_ids)
    print()