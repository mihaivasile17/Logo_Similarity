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
