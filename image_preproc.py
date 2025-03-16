import os
import numpy as np
import cv2
from tqdm import tqdm
from PIL import Image

# Cream un nou folder unde vom salva imaginile procesate
image_folder = "logos/"
processed_folder = "logos_processed/"
os.makedirs(processed_folder, exist_ok=True)

# Dimensiunea la care redimensionam imaginile
IMG_SIZE = (224, 224)

# Procesam toate imaginile
for img_name in tqdm(os.listdir(image_folder)):
    try:
        img_path = os.path.join(image_folder, img_name)
        img = Image.open(img_path).convert("RGB")
        img = img.resize(IMG_SIZE)
        img.save(os.path.join(processed_folder, img_name))
    except Exception as e:
        print(f"Eroare la procesarea {img_name}: {e}")

print("Toate imaginile au fost preprocesate și salvate!")
