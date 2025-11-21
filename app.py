# app.py
# Dog Breed Recommender — Streamlit hosted app (vertical layout, images + auto-generated videos)
# Replace or place this file in the root of your GitHub repo and include data/ folder.
 
import streamlit as st
import pandas as pd
import numpy as np
import requests
import re
import os
import tempfile
import time
from io import BytesIO
from PIL import Image as PILImage
import matplotlib.pyplot as plt
from moviepy.editor import ImageSequenceClip

# -----------------------
# Config
# -----------------------
st.set_page_config(page_title="Dog Breed Recommender", layout="centered")

DATA_DIR = "data"
BREED_CSV = os.path.join(DATA_DIR, "breed_traits.csv")
TRAIT_CSV = os.path.join(DATA_DIR, "trait_descriptions.csv")

# GitHub raw base for Dog-Breeds-Dataset
GITHUB_RAW_BASE = "https://raw.githubusercontent.com/maartenvandenbroeck/Dog-Breeds-Dataset/master/breeds"

# Max images to try per breed for video slideshow
MAX_IMAGES_PER_BREED = 5
VIDEO_FRAME_DURATION = 1.0  # seconds per image
VIDEO_FPS = 24

# -----------------------
# Utility functions
# -----------------------
@st.cache_data
def load_data():
    """Load breed_traits and trait_descriptions from local data folder."""
    breed_traits = pd.read_csv(BREED_CSV)
    trait_desc = pd.read_csv(TRAIT_CSV)
    return breed_traits, trait_desc

def normalize_breed_variants(breed_name):
    """
    Generate a list of possible GitHub folder name variants for a given breed name.
    We attempt several transformations to match the repository folder naming.
    """
    variants = []
    # Original raw (but replace NBSP with space)
    raw = breed_name.replace("\xa0", " ").strip()
    variants.append(raw)
    # Simple underscore + capitalize words
    words = raw.split()
    variants.append("_".join(word.capitalize() for word in words))
    # Replace non-alphanumeric with underscore, capitalize
    cleaned = re.sub(r"[^0-9A-Za-z() ]+", "", raw)  # remove unusual punctuation except parentheses
    cleaned2 = re.sub(r"[ ()]+", "_", cleaned).strip("_")
    variants.append(cleaned2)
    # Replace spaces by underscores preserving parentheses with underscores
    variants.append(re.sub(r"\s+", "_", raw))
    # Try capitalization with parentheses preserved and spaces->underscore
    variants.append("_".join([w.capitalize() for w in re.sub(r"\s+", " ", raw).split(" ")]))
    # Add lower-cased underscore form (some repos use lowercase)
    variants.append(cleaned2.lower())
    # Unique and preserve order
    seen = set()
    uniq = []
    for v in variants:
        if v and v not in seen:
            uniq.append(v)
            seen.add(v)
    return uniq

@st.cache_data(show_spinner=False)
def try_fetch_image_bytes(url, timeout=6):
    """Try fetching binary content from URL, return bytes or None."""
    try:
        resp = requests.get(url, timeout=timeout)
        if resp.status_code == 200:
            return resp.content
    except Exception:
        return None
    return None

@st.cache_data(show_spinner=False)
def fetch_images_for_breed(breed_name, max_images=MAX_IMAGES_PER_BREED):
    """
    Try to fetch up to max_images for the breed from GitHub.
    Returns a list of PIL images (in memory). If none, returns empty list.
    """
    images = []
    folder_variants = normalize_breed_variants(breed_name)
    exts = ["jpg", "jpeg", "png"]
    # Try numbered files 1..max_images in each variant
    for folder in folder_variants:
        for idx in range(1, max_images + 1):
            for ext in exts:
                url = f"{GITHUB_RAW_BASE}/{folder}/{idx}.{ext}"
                content = try_fetch_image_bytes(url)
                if content:
                    try:
                        img = PILImage.open(BytesIO(content)).convert("RGB")
                        images.append(img)
                        break  # found image for this idx, move to next idx
                    except Exception:
                        continue
            # stop early if reached required number
        if len(images) >= 1:
            # if we found at least one in this folder variant, keep trying indices until exhausted
            # but we also break folder variants only after we tried to collect up to max_images
            if len(images) >= max_images:
                break
            # else keep trying next idx in the same folder (loop continues)
        # If no images found for this folder at all, try next variant
    # Ensure unique images (dedupe by size+mode)
    uniq = []
    seen = set()
    for img in images:
        key = (img.size, img.mode)
        if key not in seen:
            uniq.append(img)
            seen.add(key)
    return uniq[:max_images]

def create_slideshow_video_from_pil(images, fps=VIDEO_FPS, duration_per_image=VIDEO_FRAME_DURATION):
    """
    Create a short MP4 video from a list of PIL images.
    Returns path to temporary mp4 file or None on failure.
    """
    if not images:
        return None
    tmp_dir = tempfile.mkdtemp()
    frame_paths = []
    try:
        # save frames as temporary files
        for i, img in enumerate(images):
            frame_path = os.path.join(tmp_dir, f"frame_{i:03d}.jpg")
            img.save(frame_path, format="JPEG")
            frame_paths.append(frame_path)
        # Create a moviepy clip
        durations = [duration_per_image] * len(frame_paths)
        clip = ImageSequenceClip(frame_paths, durations=durations)
        out_path = os.path.join(tmp_dir, "slideshow.mp4")
        # write_videofile can be slow; write with preset small size to save time
        clip.write_videofile(out_path, fps=fps, codec="libx264", audio=False, verbose=False, logger=None)
        return out_path
    except Exception as e:
        # Clean up on failure
        return None

def generate_breed_description_from_traits(breed_row, traits, top_n=3):
    """
    Create a short descriptive sentence highlighting top traits for the breed.
    """
    trait_scores = {t: float(breed_row.get(t, 0)) for t in traits}
    sorted_traits = sorted(trait_scores.items(), key=lambda x: x[1], reverse=True)
    top = sorted_traits[:top_n]
    phrases = []
    for t, score in top:
        # shorten trait name and include score
        short = t
        phrases.append(f"{short} ({int(score)})")
    return "Strong traits — " + ", ".join(phrases)

# -----------------------
# Load data
# -----------------------
breed_traits, trait_desc = load_data()

# Use traits from trait_desc and ignore coat ones
ignore_traits = ["Coat Type", "Coat Length"]
traits = [t for t in trait_desc["Trait"].tolist() if t not in ignore_traits]

# Ensure breed_traits has numeric columns for all traits (if missing, add zeros)
for t in traits:
    if t not in breed_traits.columns:
        breed_traits[t] = 0
breed_traits[traits] = breed_traits[traits].apply(pd.to_numeric, errors="coerce").fillna(0)

# -----------------------
# Streamlit UI
# -----------------------
st.title("🐶 Dog Breed Recommender")
st.markdown(
    "Welcome! To recommend a suitable dog breed for you, please rate the following traits. Shall we start?"
)

# start / reset controls
if "started" not in st.session_state:
    st.session_state.started = False
if "step" not in st.session_state:
    st.session_state.step = 0
if "answers" not in st.session_state:
    st.session_state.answers = {}
if "skipped" not in st.session_state:
    st.session_state.skipped = set()

col1, col2 = st.columns([3,1])
with col1:
    name = st.text_input("What's your name?", value=st.session_state.get("name", ""))
with col2:
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

# Question-by-question UI
total_traits = len(traits)
if st.session_state.step < total_traits:
    cur_trait = traits[st.session_state.step]
    st.subheader(f"**{cur_trait}**")
    # show description if available
    filtered = trait_desc[trait_desc["Trait"].str.strip().str.lower() == cur_trait.strip().lower()]
    if not filtered.empty:
        row = filtered.iloc[0]
        st.write(row["Description"])
        st.write(f"**1 = {row['Trait_1']}**, **5 = {row['Trait_5']}**")
    else:
        st.info("Description unavailable for this trait.")
    # slider with previous answer if any
    prev = st.session_state.answers.get(cur_trait, 3)
    slider = st.slider("How important is this trait to you?", 1, 5, prev, key=f"slider_{st.session_state.step}")
    # navigation buttons
    col_back, col_skip, col_next = st.columns(3)
    if col_back.button("⬅ Back"):
        if st.session_state.step > 0:
            st.session_state.step -= 1
            # preserve slider value
            st.session_state.answers[cur_trait] = slider
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
    # All questions answered — show summary
    st.success("All questions answered!")
    st.subheader("📝 Your Answers Summary")
    summary_rows = []
    for t in traits:
        if t in st.session_state.answers:
            summary_rows.append({"Trait": t, "Your Rating": st.session_state.answers[t]})
        elif t in st.session_state.skipped:
            summary_rows.append({"Trait": t, "Your Rating": "Skipped"})
        else:
            summary_rows.append({"Trait": t, "Your Rating": "Not answered"})
    summary_df = pd.DataFrame(summary_rows)
    st.table(summary_df)

    # Compute scores
    used_traits = [t for t in traits if t in st.session_state.answers]
    if not used_traits:
        st.warning("No trait rated — cannot compute recommendations. Please rate at least one trait.")
        st.stop()

    def score_row(row):
        s = 0.0
        for t in used_traits:
            s += float(row.get(t, 0)) * float(st.session_state.answers[t])
        return s

    breed_traits["Total_Score"] = breed_traits.apply(score_row, axis=1)
    top_breeds = breed_traits.sort_values(by="Total_Score", ascending=False).head(3)

    st.subheader(f"Top 3 Recommended Breeds for {user_name}:")
    for idx, row in top_breeds.iterrows():
        breed_name = row["Breed"]
        st.markdown(f"### 🐾 {breed_name} — Score: {int(row['Total_Score'])}")

        # Try fetch images from GitHub
        with st.spinner("Fetching images..."):
            images = fetch_images_for_breed(breed_name, max_images=MAX_IMAGES_PER_BREED)

        if images:
            # display first image
            st.image(images[0], use_column_width=True)
            # create video (cached to avoid re-generation)
            video_path = create_slideshow_video_from_pil(images, fps=VIDEO_FPS, duration_per_image=VIDEO_FRAME_DURATION)
            if video_path and os.path.exists(video_path):
                st.video(video_path)
            else:
                st.info("Could not create slideshow video.")
        else:
            # fallback: show GitHub folder link
            variants = normalize_breed_variants(breed_name)
            guessed_folder = variants[0].replace(" ", "_")
            url = f"https://github.com/maartenvandenbroeck/Dog-Breeds-Dataset/tree/master/breeds/{guessed_folder}"
            st.write(f"Images not found from GitHub for **{breed_name}**. You can view images here:")
            st.write(f"[Open GitHub folder for {breed_name}]({url})")

        # auto-generated short description
        desc_text = generate_breed_description_from_traits(row, traits, top_n=4)
        st.markdown("**Description:**")
        st.write(desc_text)
        st.write("---")

    # Pie chart of top breed scores
    st.subheader("📊 Score Distribution of Top Breeds")
    fig, ax = plt.subplots()
    ax.pie(top_breeds["Total_Score"], labels=top_breeds["Breed"], autopct="%1.1f%%")
    ax.set_title("Recommendation Weighting")
    st.pyplot(fig)

    # Explain logic
    st.subheader("How We Calculated These Recommendations")
    st.write("""
    - Each breed in our dataset has numeric trait values.
    - You specified how important each trait is to you (1-5).
    - For each breed we multiply the breed's trait value by your importance rating and sum across selected traits.
    - The breeds with the highest total scores are recommended.
    """)

    # Post-options
    choice = st.radio("What would you like to do next?", ["No, thank you", "Retake the test", "Download results"])
    if choice == "Retake the test":
        st.session_state.started = True
        st.session_state.step = 0
        st.session_state.answers = {}
        st.session_state.skipped = set()
        st.experimental_rerun()
    elif choice == "No, thank you":
        st.success(f"Thank you for using the Dog Breed Recommender, {user_name}! 🐾")
    elif choice == "Download results":
        # prepare CSV of top breeds and scores
        csv = top_breeds[["Breed", "Total_Score"]].to_csv(index=False)
        st.download_button("Download top breeds CSV", csv, file_name="top_breeds.csv", mime="text/csv")

