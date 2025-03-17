import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
from tqdm import tqdm

# Define image processing parameters
img_height, img_width = 224, 224
processed_folder = "logos_processed/"

# Load pre-trained ResNet50 (without top layers)
resnet_model = ResNet50(include_top=False, input_shape=(img_height, img_width, 3), pooling='avg', weights='imagenet')

# Freeze all layers (like in your Keras example)
resnet_model.trainable = False

# Function to extract features
def extract_features(image_path):
    try:
        # Load and process the image
        img = load_img(image_path, target_size=(img_height, img_width))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)  # Apply ResNet preprocessing

        # Extract features
        features = resnet_model.predict(img_array)
        return features.flatten()  # Flatten to 1D vector
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

# Process all images and store embeddings
features_dict = {}
for img_name in tqdm(os.listdir(processed_folder)):
    img_path = os.path.join(processed_folder, img_name)
    embedding = extract_features(img_path)
    if embedding is not None:
        features_dict[img_name] = embedding

# Save embeddings to a NumPy file
np.save("logo_embeddings.npy", features_dict)
print("Embeddings saved successfully!")
