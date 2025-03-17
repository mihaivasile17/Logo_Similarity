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

# Google image scraping (in case we can't find the logo by using Clearbit) - fallback method
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
