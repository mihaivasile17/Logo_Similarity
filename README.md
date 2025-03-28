# Logo Similarity  
**Veridion Internship Challenge**  

This project aims to **match and group websites** based on the similarity of their **logos**. Using **ResNet50** and **K-Means clustering**, I built a system that detects similar logos efficiently.

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
 - pip install pyarrow fastparquet - so I can read the actual file
 - after I printed the header from the file, I observed that the file is full with websites domains
 - I found Clearbit, which is offering a free API tool that can give us the logos, if it has the specific domain
 - Running the script saves the logos in the specific folder

Update: To improve the logo similarity project, I added a fallback method for downloading the logos when Clearbit Api fails. Since some domains did not return any logo by using Clearbit and I only got 799 logos initially, I implemented a method which uses Google Images. This was a success and I can confidently say that by sharing the final number of downloaded logos, which is 3415.

![results_clearbit+google](https://github.com/user-attachments/assets/6614b798-8f25-4e0a-85c4-8a4530722f8c)

![results_clearbit+google2](https://github.com/user-attachments/assets/dc9452f6-e778-4c25-b623-20ea22881228)

Logo_download_script:

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

# Google images scraping (in case I can't find the logo by using Clearbit) - fallback method
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

I managed to download 799 logos (3415 logos after I implemented the fallback method) from the given file, and now I will try to process the images:
- All the images will be converted into a resolution of 224X224
- All the images will be transformed into RGB format

![All_images_proc_saved](https://github.com/user-attachments/assets/ad422ebe-e8de-4ba5-a2a7-4cc9bd313387)

Example of image transformation:

![image_proc_example](https://github.com/user-attachments/assets/9514ca13-25fc-4b10-883f-67a68a18d0e6)

Image_processing_script:

```bash
import os
import numpy as np
import cv2
from tqdm import tqdm
from PIL import Image

# Creating a new folder where we will save the processed images
image_folder = "logos/"
processed_folder = "logos_processed/"
os.makedirs(processed_folder, exist_ok=True)

# The size we will convert the images to
IMG_SIZE = (224, 224)

# Iterating through all the images in the folder and process them
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
- Storing the vectors in a NumPy array

![extract_specs_example](https://github.com/user-attachments/assets/e6ca07d8-aaa5-4f88-9990-2c806b1e61cc)

![extract_specs_example2](https://github.com/user-attachments/assets/4ea1d4b1-8fa2-44a2-b184-bd4b65741cff)

Feature_extraction_script:

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

# Load pre-trained ResNet50
resnet_model = ResNet50(include_top=False, input_shape=(img_height, img_width, 3), pooling='avg', weights='imagenet')

# Freeze all layers
resnet_model.trainable = False

# Function to extract features
def extract_features(image_path):
    try:
        # Load and process the image
        img = load_img(image_path, target_size=(img_height, img_width))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)  # Apply ResNet processing

        # Extract features
        features = resnet_model.predict(img_array)
        return features.flatten()
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

# Saving all embeddings into a NumPy file
np.save("logo_embeddings.npy", features_dict)
print("Embeddings saved successfully!")
```

### **K-means algorithm**
Similar logo search
- First method: After obtaining the feature vectors, I used cosine distance to find similar logos
    - using this method, I could tell from the results that the classification was not great, because I saw in the same cluster, logos like mazda and infiniti which should not be together
- So, I went to try another method. I implemented a clustering algorithm, more specific K-means clustering, where I approximated a number of 500 clusters for the total of 3415 images, because I can also have a cluster with a single image inside.

![k-means_clustering](https://github.com/user-attachments/assets/d561298a-0470-433c-ab54-483b9ffe1d44)

K-means_algorithm_script:

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

---
## Final results

Main_printing_results_script:

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

Results from each cluster, sorted by cluster ID:

Old results:

![resultsold](https://github.com/user-attachments/assets/3b939770-5b4d-4428-b823-f77bee3f635e)

Final results:

![final_results](https://github.com/user-attachments/assets/98f20d94-5b17-4922-9187-02c4d24ab5cb)

![final_results2](https://github.com/user-attachments/assets/e71ee5a1-0278-4243-8087-17e43a5b0ff7)

We can see that there is room for improvements, because the results are not always as accurate as it should be:

![final_results3](https://github.com/user-attachments/assets/7962805b-b03b-4dbd-beb0-c5bfc30a5e70)