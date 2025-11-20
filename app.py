# app_clean.py
import streamlit as st
import pandas as pd
import numpy as np
import requests
import re
import os
import tempfile
from io import BytesIO
from PIL import Image as PILImage
import matplotlib.pyplot as plt
from moviepy.editor import ImageSequenceClip

st.set_page_config(page_title="Dog Breed Recommender", layout="centered")

DATA_DIR = "data"
BREED_CSV = os.path.join(DATA_DIR, "breed_traits.csv")
TRAIT_CSV = os.path.join(DATA_DIR, "trait_descriptions.csv")
GITHUB_RAW_BASE = "https://raw.githubusercontent.com/maartenvandenbroeck/Dog-Breeds-Dataset/master/breeds"

MAX_IMAGES_PER_BREED = 5
VIDEO_FRAME_DURATION = 1.0
VIDEO_FPS = 24

# -----------------------
# Utility functions
# -----------------------
@st.cache_data
def load_data():
    breed_traits = pd.read_csv(BREED_CSV)
    trait_desc = pd.read_csv(TRAIT_CSV)
    return breed_traits, trait_desc

def normalize_breed_variants(breed_name):
    variants = []
    raw = breed_name.replace("\xa0", " ").strip()
    variants.append(raw)
    words = raw.split()
    variants.append("_".join(word.capitalize() for word in words))
    cleaned = re.sub(r"[^0-9A-Za-z() ]+", "", raw)
    cleaned2 = re.sub(r"[ ()]+", "_", cleaned).strip("_")
    variants.append(cleaned2)
    variants.append(re.sub(r"\s+", "_", raw))
    variants.append("_".join([w.capitalize() for w in re.sub(r"\s+", " ", raw).split(" ")]))
    variants.append(cleaned2.lower())
    seen = set()
    uniq = []
    for v in variants:
        if v and v not in seen:
            uniq.append(v)
            seen.add(v)
    return uniq

@st.cache_data(show_spinner=False)
def try_fetch_image_bytes(url, timeout=6):
    try:
        resp = requests.get(url, timeout=timeout)
        if resp.status_code == 200:
            return resp.content
    except Exception:
        return None
    return None

@st.cache_data(show_spinner=False)
def fetch_images_for_breed(breed_name, max_images=MAX_IMAGES_PER_BREED):
    images = []
    folder_variants = normalize_breed_variants(breed_name)
    exts = ["jpg", "jpeg", "png"]
    for folder in folder_variants:
        for idx in range(1, max_images + 1):
            for ext in exts:
                url = f"{GITHUB_RAW_BASE}/{folder}/{idx}.{ext}"
                content = try_fetch_image_bytes(url)
                if content:
                    try:
                        img = PILImage.open(BytesIO(content)).convert("RGB")
                        images.append(img)
                        break
                    except Exception:
                        continue
        if len(images) >= max_images:
            break
    uniq = []
    seen = set()
    for img in images:
        key = (img.size, img.mode)
        if key not in seen:
            uniq.append(img)
            seen.add(key)
    return uniq[:max_images]

def create_slideshow_video_from_pil(images, fps=VIDEO_FPS, duration_per_image=VIDEO_FRAME_DURATION):
    if not images:
        return None
    tmp_dir = tempfile.mkdtemp()
    frame_paths = []
    try:
        for i, img in enumerate(images):
            frame_path = os.path.join(tmp_dir, f"frame_{i:03d}.jpg")
            img.save(frame_path, format="JPEG")
            frame_paths.append(frame_path)
        durations = [duration_per_image] * len(frame_paths)
        clip = ImageSequenceClip(frame_paths, durations=durations)
        out_path = os.path.join(tmp_dir, "slideshow.mp4")
        clip.write_videofile(out_path, fps=fps, codec="libx264", audio=False, verbose=False, logger=None)
        return out_path
    except Exception:
        return None

def generate_breed_description_from_traits(breed_row, traits, top_n=3):
    trait_scores = {t: float(breed_row.get(t, 0)) for t in traits}
    sorted_traits = sorted(trait_scores.items(), key=lambda x: x[1], reverse=True)
    top = sorted_traits[:top_n]
    phrases = [f"{t} ({int(score)})" for t, score in top]
    return "Strong traits — " + ", ".join(phrases)

# -----------------------
# Load data
# -----------------------
breed_traits, trait_desc = load_data()
ignore_traits = ["Coat Type", "Coat Length"]
traits = [t for t in trait_desc["Trait"].tolist() if t not in ignore_traits]
for t in traits:
    if t not in breed_traits.columns:
        breed_traits[t] = 0
breed_traits[traits] = breed_traits[traits].apply(pd.to_numeric, errors="coerce").fillna(0)

# -----------------------
# Streamlit UI
# -----------------------
st.title("🐶 Dog Breed Recommender")

# --- [User inputs and trait rating UI remain unchanged] ---

# After computing top breeds
try:
    fig, ax = plt.subplots()
    ax.pie(top_breeds["Total_Score"], labels=top_breeds["Breed"], autopct="%1.1f%%")
    ax.set_title("Recommendation Weighting")
    st.pyplot(fig)
except Exception:
    st.info("Could not generate score chart (matplotlib issue).")


