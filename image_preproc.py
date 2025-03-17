import os
import numpy as np
import cv2
from tqdm import tqdm
from PIL import Image

# We create a new folder where we will save the processed images
image_folder = "logos/"
processed_folder = "logos_processed/"
os.makedirs(processed_folder, exist_ok=True)

# The size we will convert the images to
IMG_SIZE = (224, 224)

# We iterate through all the images in the folder and process them
for img_name in tqdm(os.listdir(image_folder)):
    try:
        img_path = os.path.join(image_folder, img_name)
        img = Image.open(img_path).convert("RGB")
        img = img.resize(IMG_SIZE)
        img.save(os.path.join(processed_folder, img_name))
    except Exception as e:
        print(f"Error at processing {img_name}: {e}")

print("All images saved and processed")
