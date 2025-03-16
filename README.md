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
### **Install Required Libraries**
```bash
pip install pyarrow fastparquet requests opencv-python numpy torch torchvision scikit-learn fastapi uvicorn streamlit tqdm pillow

```bash
pip install pyarrow fastparquet requests opencv-python numpy torch torchvision scikit-learn fastapi uvicorn streamlit tqdm pillow
