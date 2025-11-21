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
from urllib.parse import quote

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
    # load CSVs and do basic cleaning (remove stray weird characters)
    breed_traits = pd.read_csv(BREED_CSV)
    trait_desc = pd.read_csv(TRAIT_CSV)
    # normalize column strings
    for df in (breed_traits, trait_desc):
        for c in df.select_dtypes(include=[object]).columns:
            df[c] = df[c].astype(str).str.replace('\u00a0', ' ', regex=False).str.replace('Â', '', regex=False)
    return breed_traits, trait_desc


def normalize_breed_variants(breed_name):
    """Return a list of plausible folder/filename variants used in many GitHub image datasets.
    Also remove unsafe filename characters and return url-quoteable strings where needed."""
    if not isinstance(breed_name, str):
        breed_name = str(breed_name)
    raw = breed_name.replace("\xa0", " ").strip()
    raw = raw.replace('Â', '')
    variants = []
    variants.append(raw)
    words = raw.split()
    variants.append("_".join(word.capitalize() for word in words))
    cleaned = re.sub(r"[^0-9A-Za-z() ]+", "", raw)
    cleaned2 = re.sub(r"[ ()]+", "_", cleaned).strip("_")
    variants.append(cleaned2)
    variants.append(re.sub(r"\s+", "_", raw))
    variants.append("_".join([w.capitalize() for w in re.sub(r"\s+", " ", raw).split(" ")]))
    variants.append(cleaned2.lower())
    # also add url-quoted versions (for display as direct links)
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
        if resp.status_code == 200 and 'image' in resp.headers.get('content-type', ''):
            return resp.content
    except Exception:
        return None
    return None


@st.cache_data(show_spinner=False)
def fetch_images_for_breed(breed_name, max_images=MAX_IMAGES_PER_BREED):
    """Try multiple filename/folder variants and common extensions. Returns list of PIL images.
    If none found returns empty list.
    """
    images = []
    folder_variants = normalize_breed_variants(breed_name)
    exts = ["jpg", "jpeg", "png"]
    for folder in folder_variants:
        for idx in range(1, max_images + 1):
            for ext in exts:
                # use quoted path components to handle spaces/special chars
                folder_q = quote(folder, safe='')
                url = f"{GITHUB_RAW_BASE}/{folder_q}/{idx}.{ext}"
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
    # deduplicate by size+mode
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
    # take numeric traits if possible
    trait_scores = {}
    for t in traits:
        try:
            trait_scores[t] = float(breed_row.get(t, 0))
        except Exception:
            trait_scores[t] = 0.0
    sorted_traits = sorted(trait_scores.items(), key=lambda x: x[1], reverse=True)
    top = sorted_traits[:top_n]
    phrases = [f"{t} ({int(score)})" for t, score in top]
    return "Strong traits — " + ", ".join(phrases)


# -----------------------
# Load data
# -----------------------
breed_traits, trait_desc = load_data()
ignore_traits = ["Coat Type", "Coat Length"]
# ensure the trait list uses the Trait column from trait_desc
traits = [t for t in trait_desc["Trait"].tolist() if t not in ignore_traits]
# add missing columns with zeros
for t in traits:
    if t not in breed_traits.columns:
        breed_traits[t] = 0
# coerce numeric trait columns
for t in traits:
    try:
        breed_traits[t] = pd.to_numeric(breed_traits[t], errors="coerce").fillna(0)
    except Exception:
        breed_traits[t] = 0

# -----------------------
# Streamlit UI
# -----------------------
st.title("🐶 Dog Breed Recommender")
st.markdown("Welcome! Rate traits to get your top recommended dog breeds.")

# start/reset/exit controls
if "started" not in st.session_state:
    st.session_state.started = False
if "step" not in st.session_state:
    st.session_state.step = 0
if "answers" not in st.session_state:
    st.session_state.answers = {}
if "skipped" not in st.session_state:
    st.session_state.skipped = set()
if "name" not in st.session_state:
    st.session_state.name = ""
if "results_shown" not in st.session_state:
    st.session_state.results_shown = False

# Name input
name = st.text_input("What's your name?", value=st.session_state.get("name", ""))
cols_ctrl = st.columns([1,1,1])
with cols_ctrl[0]:
    if st.button("Start" if not st.session_state.started else "Restart"):
        st.session_state.started = True
        st.session_state.step = 0
        st.session_state.answers = {}
        st.session_state.skipped = set()
        st.session_state.name = name
        st.session_state.results_shown = False
        st.experimental_rerun()
with cols_ctrl[1]:
    if st.button("Exit"):
        user = st.session_state.get("name", name) or "friend"
        st.success(f"Thank you, {user}! Thanks for trying the Dog Breed Recommender. Have a great day 🐾")
        st.stop()
with cols_ctrl[2]:
    st.write("")

if not st.session_state.started:
    st.info("Click **Start** when ready. You can change your name before starting.")
    st.stop()

user_name = st.session_state.get("name", name)
if user_name:
    st.write(f"Nice to meet you, **{user_name}**! We'll ask about each trait — rate importance 1 (low) to 5 (high).")

# Question loop with progress and trait description/rating labels
total_traits = len(traits)
if st.session_state.step < total_traits:
    cur_idx = st.session_state.step
    cur_trait = traits[cur_idx]

    # progress
    st.markdown(f"**Question {cur_idx+1} of {total_traits}**")
    progress = (cur_idx) / max(1, total_traits)
    st.progress(progress)

    # trait description and rating criteria
    trow = trait_desc[trait_desc['Trait'] == cur_trait]
    trait_description = ""
    trait_low = "1"
    trait_high = "5"
    if not trow.empty:
        trow = trow.iloc[0]
        trait_description = trow.get('Description', '')
        trait_low = trow.get('Trait_1', '1')
        trait_high = trow.get('Trait_5', '5')

    st.subheader(f"{cur_trait}")
    with st.expander("Trait description and rating criteria (click to expand)"):
        if trait_description:
            st.write(trait_description)
        st.write(f"**Rating scale:** 1 = *{trait_low}*  —  5 = *{trait_high}*")

    # slider
    slider = st.slider("How important is this trait to you?", 1, 5, st.session_state.answers.get(cur_trait, 3), key=f"slider_{cur_idx}")

    # navigation
    col_back, col_skip, col_next = st.columns(3)
    if col_back.button("⬅ Back"):
        st.session_state.answers[cur_trait] = slider
        if st.session_state.step > 0:
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
    # Results
    st.success("All questions answered!")
    used_traits = [t for t in traits if t in st.session_state.answers]
    if not used_traits:
        st.warning("No traits rated. Please rate at least one trait.")
        st.stop()

    # compute scores (weighted sum where user importance multiplies breed trait)
    def score_row(row):
        s = 0.0
        for t in used_traits:
            try:
                s += float(row.get(t, 0)) * float(st.session_state.answers[t])
            except Exception:
                pass
        return s

    breed_traits['Total_Score'] = breed_traits.apply(score_row, axis=1)
    top_breeds = breed_traits.sort_values(by='Total_Score', ascending=False).head(3)

    st.markdown("### My top 3 recommendations are...")

    # show results with intro phrase and images (+ fallback github url suggestions)
    for idx, row in top_breeds.iterrows():
        breed_name = row['Breed']
        st.markdown(f"### 🐾 {breed_name} — Score: {int(row['Total_Score'])}")

        intro = f"This dog has the following traits which {user_name or 'you'} values the most or that suit your needs:"
        st.markdown(f"**{intro}**")

        # top matching traits (closest to user's weighted preference)
        diffs = []
        for t in used_traits:
            try:
                diffs.append((t, abs(float(row.get(t, 0)) - float(st.session_state.answers[t])), row.get(t, 0)))
            except Exception:
                pass
        best = sorted(diffs, key=lambda x: x[1])[:5]
        for t, d, br in best:
            # get description of trait if available
            trow = trait_desc[trait_desc['Trait'] == t]
            desc = trow.iloc[0]['Description'] if not trow.empty else ''
            st.write(f"- **{t}**: breed = {br}, your preference = {st.session_state.answers[t]}\n  \n    {desc}")

        # images
        images = fetch_images_for_breed(breed_name, max_images=MAX_IMAGES_PER_BREED)
        if images:
            try:
                st.image(images[0], use_column_width=True, caption=breed_name)
            except Exception:
                st.write("Image could not be rendered here — open the raw image URL in your browser.")
        else:
            st.info(f"Images not found for {breed_name}.")
            # suggest example GitHub raw URLs using normalized variants
            variants = normalize_breed_variants(breed_name)
            st.write("Possible GitHub raw URLs to try (copy-paste into browser):")
            for v in variants[:6]:
                v_q = quote(v, safe='')
                example = f"{GITHUB_RAW_BASE}/{v_q}/1.jpg"
                st.write(f"- {example}")

        desc_text = generate_breed_description_from_traits(row, traits, top_n=4)
        # prepend required phrase
        final_desc = f"This dog has the following traits which {user_name or 'you'} values the most: {desc_text}"
        st.markdown("**Description:**")
        st.write(final_desc)
        st.write("---")

    # pie chart
    try:
        fig, ax = plt.subplots()
        ax.pie(top_breeds['Total_Score'], labels=top_breeds['Breed'], autopct='%1.1f%%')
        ax.set_title('Recommendation Weighting')
        st.pyplot(fig)
    except Exception:
        st.info("Could not generate score chart.")

    # After-results chat-like options
    st.markdown("---")
    st.markdown("### What would you like to do next?")
    choice = st.radio("Choose an option", options=["Exit", "Retake the quiz", "Ask for explanation of results"], index=0)

    if st.button("Confirm choice"):
        if choice == "Exit":
            user = user_name or "friend"
            st.success(f"Thank you, {user}! Glad I could help. 🐶")
            st.stop()
        elif choice == "Retake the quiz":
            st.session_state.started = True
            st.session_state.step = 0
            st.session_state.answers = {}
            st.session_state.skipped = set()
            st.session_state.results_shown = False
            st.experimental_rerun()
        elif choice == "Ask for explanation of results":
            st.header("Explanation for recommendations")
            for idx, row in top_breeds.iterrows():
                breed_name = row['Breed']
                st.subheader(breed_name)
                st.write("Top matching traits and descriptions:")
                diffs = []
                for t in used_traits:
                    try:
                        diffs.append((t, abs(float(st.session_state.answers[t]) - float(row.get(t,0))), row.get(t,0)))
                    except Exception:
                        pass
                diffs_sorted = sorted(diffs, key=lambda x: x[1])
                for t, diff, br_val in diffs_sorted[:5]:
                    trow = trait_desc[trait_desc['Trait'] == t]
                    desc = trow.iloc[0]['Description'] if not trow.empty else ''
                    st.write(f"- **{t}**: breed = {br_val}, you = {st.session_state.answers[t]} (diff {diff}). {desc}")
            st.info("If you'd like more detail or a personalized walk-through, choose 'Retake the quiz' or 'Exit' above.")

    # bottom controls
    bottom_col1, bottom_col2 = st.columns(2)
    with bottom_col1:
        if st.button("Restart Quiz (bottom)"):
            st.session_state.started = True
            st.session_state.step = 0
            st.session_state.answers = {}
            st.session_state.skipped = set()
            st.experimental_rerun()
    with bottom_col2:
        if st.button("Exit (bottom)"):
            user = user_name or "friend"
            st.success(f"Thank you, {user} — come back any time!")
            st.stop()

