# Logo Similarity  
**Veridion Internship Challenge**  

This project aims to **match and group websites** based on the similarity of their **logos**. Using **ResNet50** and **K-Means clustering**, we built a system that detects similar logos efficiently.

---

## Project Overview
 - Extract logos from **company domains**  
 - Preprocess images (**resize, RGB conversion**)  
 - Extract **feature vectors** using **ResNet50**  
 - Find similar logos using **cosine similarity & clustering**  
 - Build an **API & web app** for easy testing  

---

## Setup & Dependencies  
### **Install required libraries**
```bash
pip install pyarrow fastparquet requests opencv-python numpy torch torchvision scikit-learn fastapi uvicorn streamlit tqdm pillow
```

### **Logo download**
Am inceput prin a incerca sa accesez fisierul de logo-uri: logos.snappy.parquet 
 - pip install pyarrow fastparquet - pentru a putea citi din fisierul respectiv
 - dupa rularea fisierul de logo-uri, observam ca acesta nu contine imagini ci doar domeniile web ale companiilor
 - Clearbit oferă un API public gratuit ce permite obținerea logo-ului doar pe baza domeniului
 - rulam script-ul creat pentru a salva logo-urile ca png prin intermediul Clearbit

```bash
import os
import pandas as pd
import requests
from PIL import Image
from io import BytesIO

# Incarcam fisierul parquet ce contine logo-urile
file_path = "logos.snappy.parquet"
df = pd.read_parquet(file_path)

# Cream un folder nou in care vom salva png-urile
output_dir = "logos"
os.makedirs(output_dir, exist_ok=True)

# Descarcam logo-urile
for index, row in df.iterrows():
    domain = row["domain"]
    logo_url = f"https://logo.clearbit.com/{domain}"

    try:
        response = requests.get(logo_url, timeout=5)
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content))
            img = img.convert("RGB")
            img.save(os.path.join(output_dir, f"{domain}.png"))
            print(f"Downloaded: {domain}.png")
        else:
            print(f"No logo for {domain}")
    except Exception as e:
        print(f"Error for {domain}: {e}")
```

### **Image processing**

Am reusit sa downloadez 799 de logo-uri din fisierul dat, iar acum voi incerca sa preprocesez imaginile
 - vom converti toate pozele la o rezolutie de 224X224
 - transformam toate imaginile in format rgb

```bash
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

```

### **Extract specifications**
Extragerea caracteristicilor cu ResNet50 (model CNN pre-antrenat)
 - acest model ResNet50, trece imaginile prin el și obține vectori de caracteristici pentru fiecare logo
 - Store the embeddings in a NumPy array

```bash
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
        # Load and preprocess the image
        img = load_img(image_path, target_size=(img_height, img_width))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Expand dims for batch size
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

```

### **K-means algorithm**
Căutare de logo-uri similare
 - prima testare: după ce avem vectorii de caracteristici, putem folosi distanța cosinus pentru a găsi logo-uri similare
 - a doua testare: implementam un alg de clusteres, K-means, unde aproximam un numar de 200 de clustere

```bash
from sklearn.cluster import KMeans
import numpy as np

# Load embeddings
features = np.load("logo_embeddings.npy", allow_pickle=True).item()

# Convert embeddings to a matrix
img_names = list(features.keys())
embeddings = np.array(list(features.values()))

# Apply K-Means clustering
num_clusters = 200
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
labels = kmeans.fit_predict(embeddings)

# Organize results into clusters
clusters = {}
for img_idx, cluster_id in enumerate(labels):
    if cluster_id not in clusters:
        clusters[cluster_id] = []
    clusters[cluster_id].append(img_names[img_idx])

# Save cluster results
np.save("logo_clusters.npy", clusters)
print("K-Means clustering completed!")

```

### **Final results**
Afisarea rezultatelor din clustere:

```bash
import numpy as np

# Load cluster results
clusters = np.load("logo_clusters.npy", allow_pickle=True).item()

# Print results
print("Grouped Websites by Logo Similarity (K-Means with 200 Clusters):")
for cluster_id, websites in clusters.items():
    print(f"Cluster {cluster_id}: {websites}")

```

---
## Results

![results](https://github.com/user-attachments/assets/3d075973-fcf6-4def-bf5b-9408d4cba6ba)
