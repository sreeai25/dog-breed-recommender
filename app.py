# app.py
# Dog Breed Recommender — Streamlit hosted app (images + optional videos)

import streamlit as st
st.set_page_config(page_title="Dog Breed Recommender", layout="centered")  # MUST be first Streamlit call

import pandas as pd
import numpy as np
import requests
import re
import os
import tempfile
from io import BytesIO
from PIL import Image as PILImage
import matplotlib.pyplot as plt

# -----------------------
# Optional MoviePy import (fallback if unavailable)
# -----------------------
try:
    from moviepy.editor import ImageSequenceClip
    moviepy_available = True
except ModuleNotFoundError:
    moviepy_available = False
    st.warning("MoviePy not available — slideshow videos will be skipped.")

# -----------------------
# Config
# -----------------------
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
    if not moviepy_available or not images:
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
st.markdown("Welcome! Rate traits to get your top recommended dog breeds.")

# start/reset controls
if "started" not in st.session_state:
    st.session_state.started = False
if "step" not in st.session_state:
    st.session_state.step = 0
if "answers" not in st.session_state:
    st.session_state.answers = {}
if "skipped" not in st.session_state:
    st.session_state.skipped = set()

name = st.text_input("What's your name?", value=st.session_state.get("name", ""))
if st.button("Start" if not st.session_state.started else "Restart"):
    st.session_state.started = True
    st.session_state.step = 0
    st.session_state.answers = {}
    st.session_state.skipped = set()
    st.session_state.name = name

if not st.session_state.started:
    st.info("Click **Start** when ready. You can change your name before starting.")
    st.stop()

user_name = st.session_state.get("name", name)
if user_name:
    st.write(f"Nice to meet you, **{user_name}**! We'll ask about each trait — rate importance 1 (low) to 5 (high).")

# Question loop
total_traits = len(traits)
if st.session_state.step < total_traits:
    cur_trait = traits[st.session_state.step]
    st.subheader(f"**{cur_trait}**")
    slider = st.slider("How important is this trait?", 1, 5, st.session_state.answers.get(cur_trait, 3), key=f"slider_{st.session_state.step}")
    col_back, col_skip, col_next = st.columns(3)
    if col_back.button("⬅ Back"):
        if st.session_state.step > 0:
            st.session_state.answers[cur_trait] = slider
            st.session_state.step -= 1
            st.experimental_rerun()
    if col_skip.button("Skip ❌"):
        st.session_state.skipped.add(cur_trait)
        if cur_trait in st.session_state.answers:
            del st.session_state.answers[cur_trait]
        st.session_state.step += 1
        st.experimental_rerun()
    if col_next.button("Next ➡"):
        st.session_state.answers[cur_trait] = slider
        if cur_trait in st.session_state.skipped:
            st.session_state.skipped.remove(cur_trait)
        st.session_state.step += 1
        st.experimental_rerun()

else:
    st.success("All questions answered!")
    # compute top breeds
    used_traits = [t for t in traits if t in st.session_state.answers]
    if not used_traits:
        st.warning("No traits rated.")
        st.stop()
    def score_row(row):
        return sum(float(row.get(t,0)) * st.session_state.answers[t] for t in used_traits)
    breed_traits["Total_Score"] = breed_traits.apply(score_row, axis=1)
    top_breeds = breed_traits.sort_values(by="Total_Score", ascending=False).head(3)

    for idx, row in top_breeds.iterrows():
        breed_name = row["Breed"]
        st.markdown(f"### 🐾 {breed_name} — Score: {int(row['Total_Score'])}")
        images = fetch_images_for_breed(breed_name, max_images=MAX_IMAGES_PER_BREED)
        if images:
            st.image(images[0], use_column_width=True)
            if moviepy_available:
                video_path = create_slideshow_video_from_pil(images)
                if video_path and os.path.exists(video_path):
                    st.video(video_path)
                else:
                    st.info("Could not create slideshow video.")
            else:
                st.info("Video generation skipped because MoviePy is not installed.")
        else:
            st.info(f"Images not found for {breed_name}.")

        desc_text = generate_breed_description_from_traits(row, traits, top_n=4)
        st.markdown("**Description:**")
        st.write(desc_text)
        st.write("---")

    # Pie chart of top breed scores
    try:
        fig, ax = plt.subplots()
        ax.pie(top_breeds["Total_Score"], labels=top_breeds["Breed"], autopct="%1.1f%%")
        ax.set_title("Recommendation Weighting")
        st.pyplot(fig)
    except Exception:
        st.info("Could not generate score chart.")

