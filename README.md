# Logo Similarity  
**Veridion Internship Challenge**  

This project aims to **match and group websites** based on the similarity of their **logos**. Using **ResNet50** and **K-Means clustering**, we built a system that detects similar logos efficiently.

---

## Project overview
 - Extract logos from **company domains**  
 - Preprocess images (**resize, RGB conversion**)  
 - Extract **feature vectors** using **ResNet50**  
 - Find similar logos using **cosine similarity** (first version) & **clustering** (curent version)

---

## Setup & Dependencies  
### **Install required libraries**
```bash
pip install pyarrow fastparquet requests opencv-python numpy torch torchvision scikit-learn fastapi uvicorn streamlit tqdm pillow
```

### **Logo download**
I started by trying to figure out what exactly is in the logos.snappy.parquet file:
 - pip install pyarrow fastparquet - so we can read in the actual file
 - after we print the header from the file, we observe that the file is full with websites domains
 - I found Clearbit, which is offering a free API tool that can give us the logos, if we give the specific domain
 - we run the script and we will save the logos as pngs in the logos folder

Update: To improve the logo similarity project, I added a fallback method for downloading the logos when Clearbit Api fails. Since some domains did not return a logo by using Clearbit, and we only got 799 logos initially, I implemented a method which uses Google Images. This was a success and I can confidently say that by sharing the final number of downloaded logos, which is 3415.

```bash
import os
import pandas as pd
import requests
from PIL import Image
from io import BytesIO
from bs4 import BeautifulSoup
import urllib.parse

# Load the parquet file that contains domains
file_path = "logos.snappy.parquet"
df = pd.read_parquet(file_path)

# Create a folder to save logos
output_dir = "logos"
os.makedirs(output_dir, exist_ok=True)

# Clearbit API (first tool to download the logos)
def fetch_clearbit_logo(domain):
    url = f"https://logo.clearbit.com/{domain}"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            return response.content
    except requests.RequestException:
        pass
    return None

# Google image scraping (in case I can't find the logo by using Clearbit) - fallback method
def fetch_google_logo(domain):
    try:
        search_url = f"https://www.google.com/search?tbm=isch&q={urllib.parse.quote(domain + ' logo')}"
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(search_url, headers=headers)

        soup = BeautifulSoup(response.text, "html.parser")
        img_tags = soup.find_all("img")

        for img_tag in img_tags:
            img_url = img_tag.get("src")
            if img_url.startswith("data:image"):
                continue
            if img_url.startswith("//"):
                img_url = "https:" + img_url
            # Check if it is a valid image
            if img_url.startswith("http"):
                try:
                    img_response = requests.get(img_url, timeout=5)
                    if img_response.status_code == 200:
                        return img_response.content
                except requests.RequestException:
                    continue  # Try next image
    except requests.RequestException:
        pass
    return None  # If no valid image is found

# Saving the images
def save_logo(domain, logo):
    if logo:
        try:
            image = Image.open(BytesIO(logo))
            image = image.convert("RGB")
            image.save(os.path.join(output_dir, f"{domain}.png"))
            print(f"Saved: {domain}.png")
        except Exception as e:
            print(f"Error saving {domain}: {e}")
    else:
        print(f"No logo found for {domain}")

# Loop through the domains and download the logos
for index, row in df.iterrows():
    domain = row["domain"]
    logo = fetch_clearbit_logo(domain)
    if logo is None:
        print(f"Clearbit failed for {domain}, trying Google Images")
        logo = fetch_google_logo(domain)
    save_logo(domain, logo)
```

### **Image processing**

I managed to download 799 logos (3415 logos after implemented the fallback method) from the given file, and now I will try to process the images:
- we will convert all the pictures to a resolution of 224X224
- we will transform all the images into rgb format

```bash
import os
import numpy as np
import cv2
from tqdm import tqdm
from PIL import Image

# I create a new folder where we will save the processed images
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
```

### **Extract specifications**
Feature extraction with ResNet50 (pre-trained CNN model)
- I decided to use Keras ResNet50 for feature extraction instead of PyTorch, because I had better results with it
- ResNet50 model processes images and extracts feature vectors for each logo
- We store the vectors in a NumPy array

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
```

### **K-means algorithm**
Similar logo search
- First method: After obtaining the feature vectors, I used cosine distance to find similar logos
    - using this method, I could tell from the results that the classification was not great, because I saw in the same cluster, car logos like mazda and infiniti which should not be together
- So, I went to try another method. I implemented a clustering algorithm, more specific K-means clustering, where we approximate a number of 500 clusters for the total of 3415 images, because we can have a cluster with a single image inside.

```bash
from sklearn.cluster import KMeans
import numpy as np

# Load embeddings
features = np.load("logo_embeddings.npy", allow_pickle=True).item()

# Convert embeddings to a matrix
img_names = list(features.keys())
embeddings = np.array(list(features.values()))

# Apply K-Means clustering
num_clusters = 500
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
Results from each cluster, sorted by cluster ID:

```bash
import numpy as np

# Load cluster results
clusters = np.load("logo_clusters.npy", allow_pickle=True).item()

# Sort clusters by cluster ID
sorted_clusters = sorted(clusters.items(), key=lambda x: x[0])

# Print results
print("Grouped websites by logo similarity:")
for cluster_id, websites in sorted_clusters:
    print(f"Cluster {cluster_id} ({len(websites)} logos): {websites}")
```

---
## Final results

![results](https://github.com/user-attachments/assets/3d075973-fcf6-4def-bf5b-9408d4cba6ba)
