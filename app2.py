# app_chat.py
# Chat-style version of Dog Breed Recommender (Option A: clean ChatGPT-style bubbles)
# Based on original uploaded file: /mnt/data/app.py.txt
# Original file reference: /mnt/data/app.py.txt

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
st.set_page_config(page_title="Dog Breed Recommender — Chat", layout="centered")

DATA_DIR = "data"
BREED_CSV = os.path.join(DATA_DIR, "breed_traits.csv")
TRAIT_CSV = os.path.join(DATA_DIR, "trait_descriptions.csv")

GITHUB_RAW_BASE = "https://raw.githubusercontent.com/maartenvandenbroeck/Dog-Breeds-Dataset/master/breeds"
MAX_IMAGES_PER_BREED = 5
VIDEO_FRAME_DURATION = 1.0
VIDEO_FPS = 24

# -----------------------
# Utility functions (kept from original)
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
    phrases = []
    for t, score in top:
        short = t
        phrases.append(f"{short} ({int(score)})")
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
# Chat-style UI helpers
# -----------------------
CHAT_CSS = '''
<style>
.chat-container{width:100%;max-width:900px;margin:0 auto;}
.bubble{padding:12px 16px;border-radius:14px;display:inline-block;max-width:80%;margin-bottom:8px;}
.bot{background:#f1f3f5;color:#000;border-radius:14px;align-items:flex-start}
.user{background:#ffffff;color:#000;border-radius:14px;align-items:flex-end;box-shadow:0 1px 2px rgba(0,0,0,0.06)}
.row{display:flex;}
.row.bot-row{justify-content:flex-start;}
.row.user-row{justify-content:flex-end;}
.small{font-size:0.85rem;opacity:0.8;margin-top:4px}
.chat-box{height:520px;overflow:auto;padding:12px;border-radius:8px;border:1px solid #eee;background:transparent}
.meta{font-size:0.8rem;color:#666;margin-bottom:6px}
</style>
'''


def add_bot_message(text):
    st.session_state.messages.append({"role": "bot", "text": text})


def add_user_message(text):
    st.session_state.messages.append({"role": "user", "text": text})


def render_messages():
    st.markdown(CHAT_CSS, unsafe_allow_html=True)
    container = st.container()
    with container:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        # chat box
        st.markdown('<div class="chat-box" id="chatbox">', unsafe_allow_html=True)
        for m in st.session_state.messages:
            role = m.get("role")
            text = m.get("text")
            if role == "bot":
                st.markdown(f"<div class='row bot-row'><div class='bubble bot'>{text}</div></div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='row user-row'><div class='bubble user'>{text}</div></div>", unsafe_allow_html=True)
        st.markdown('</div></div>', unsafe_allow_html=True)
    # scroll to bottom using javascript
    st.markdown("<script>var cb = document.getElementById('chatbox'); if(cb) cb.scrollTop = cb.scrollHeight;</script>", unsafe_allow_html=True)

# -----------------------
# Initialize session state
# -----------------------
if 'started' not in st.session_state:
    st.session_state.started = False
if 'step' not in st.session_state:
    st.session_state.step = 0
if 'answers' not in st.session_state:
    st.session_state.answers = {}
if 'skipped' not in st.session_state:
    st.session_state.skipped = set()
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'name' not in st.session_state:
    st.session_state.name = ''

# -----------------------
# Top-level controls
# -----------------------
st.title("🐶 Dog Breed Recommender — Chat")
col1, col2 = st.columns([3,1])
with col1:
    name = st.text_input("What's your name?", value=st.session_state.get("name", ""))
with col2:
    if st.button("Start" if not st.session_state.started else "Restart"):
        st.session_state.started = True
        st.session_state.step = 0
        st.session_state.answers = {}
        st.session_state.skipped = set()
        st.session_state.messages = []
        st.session_state.name = name
        # initial greeting in chat style
        add_bot_message("Hi! I'll ask a few questions to recommend a dog breed for you. Rate each trait 1 (low) to 5 (high). Ready?")
        # push first trait prompt
        if traits:
            cur = traits[0]
            # include short description if available
            filtered = trait_desc[trait_desc['Trait'].str.strip().str.lower() == cur.strip().lower()]
            if not filtered.empty:
                row = filtered.iloc[0]
                add_bot_message(f"**{cur}** — {row['Description']}\n\n**1 = {row['Trait_1']}**, **5 = {row['Trait_5']}**")
            else:
                add_bot_message(f"**{cur}**")
        st.experimental_rerun()

if not st.session_state.started:
    st.info("Click **Start** to begin the chat-style quiz. You can change your name before starting.")
    st.stop()

user_name = st.session_state.get('name', name)
if user_name:
    st.markdown(f"**Nice to meet you, {user_name}!**")

# render the chat messages
render_messages()

# Show interactive controls for the current question below the chat box (styled as a chat input area)
if st.session_state.step < len(traits):
    cur_trait = traits[st.session_state.step]

    st.markdown("---")
    st.markdown(f"### Question {st.session_state.step+1} of {len(traits)}")

    # description
    filtered = trait_desc[trait_desc['Trait'].str.strip().str.lower() == cur_trait.strip().lower()]
    if not filtered.empty:
        row = filtered.iloc[0]
        desc = f"{row['Description']}\n\n1 = {row['Trait_1']}, 5 = {row['Trait_5']}"
    else:
        desc = ""

    # show bot prompt above controls (so user sees what's being asked)
    st.markdown(f"**{cur_trait}** — {desc}")

    # slider and navigation in a form to avoid multiple triggers
    with st.form(key=f"form_{st.session_state.step}"):
        prev = st.session_state.answers.get(cur_trait, 3)
        rating = st.slider("How important is this trait to you?", 1, 5, prev)
        cols = st.columns([1,1,1])
        back = cols[0].form_submit_button("⬅ Back")
        skip = cols[1].form_submit_button("Skip ❌")
        nxt = cols[2].form_submit_button("Next ➡")

    # handle form actions
    if back:
        # save current slider as draft
        st.session_state.answers[cur_trait] = rating
        if st.session_state.step > 0:
            st.session_state.step -= 1
            # show a user message for going back
            add_user_message(f"(went back) {rating}")
            # push bot message for the new current trait
            new_trait = traits[st.session_state.step]
            filtered = trait_desc[trait_desc['Trait'].str.strip().str.lower() == new_trait.strip().lower()]
            if not filtered.empty:
                r = filtered.iloc[0]
                add_bot_message(f"**{new_trait}** — {r['Description']}\n\n**1 = {r['Trait_1']}**, **5 = {r['Trait_5']}**")
            else:
                add_bot_message(f"**{new_trait}**")
            st.experimental_rerun()

    if skip:
        st.session_state.skipped.add(cur_trait)
        if cur_trait in st.session_state.answers:
            del st.session_state.answers[cur_trait]
        add_user_message("Skipped")
        st.session_state.step += 1
        # append next bot message automatically
        if st.session_state.step < len(traits):
            next_trait = traits[st.session_state.step]
            filtered = trait_desc[trait_desc['Trait'].str.strip().str.lower() == next_trait.strip().lower()]
            if not filtered.empty:
                r = filtered.iloc[0]
                add_bot_message(f"**{next_trait}** — {r['Description']}\n\n**1 = {r['Trait_1']}**, **5 = {r['Trait_5']}**")
            else:
                add_bot_message(f"**{next_trait}**")
        st.experimental_rerun()

    if nxt:
        st.session_state.answers[cur_trait] = rating
        if cur_trait in st.session_state.skipped:
            st.session_state.skipped.remove(cur_trait)
        add_user_message(str(rating))
        st.session_state.step += 1
        # append next bot message automatically
        if st.session_state.step < len(traits):
            next_trait = traits[st.session_state.step]
            filtered = trait_desc[trait_desc['Trait'].str.strip().str.lower() == next_trait.strip().lower()]
            if not filtered.empty:
                r = filtered.iloc[0]
                add_bot_message(f"**{next_trait}** — {r['Description']}\n\n**1 = {r['Trait_1']}**, **5 = {r['Trait_5']}**")
            else:
                add_bot_message(f"**{next_trait}**")
        else:
            add_bot_message("Thanks — that's all the questions. I'll compute recommendations now.")
        st.experimental_rerun()

else:
    # finished: show summary and recommendations in chat
    add_bot_message("All done! Here's a brief summary of your answers:")
    summary_lines = []
    for t in traits:
        if t in st.session_state.answers:
            summary_lines.append(f"{t}: {st.session_state.answers[t]}")
        elif t in st.session_state.skipped:
            summary_lines.append(f"{t}: Skipped")
        else:
            summary_lines.append(f"{t}: Not answered")
    add_bot_message("\n".join(summary_lines[:20]))

    # compute recommendations
    used_traits = [t for t in traits if t in st.session_state.answers]
    if not used_traits:
        add_bot_message("No traits rated — please retake the quiz if you'd like recommendations.")
        render_messages()
        st.stop()

    def score_row(row):
        s = 0.0
        for t in used_traits:
            s += float(row.get(t, 0)) * float(st.session_state.answers[t])
        return s

    breed_traits['Total_Score'] = breed_traits.apply(score_row, axis=1)
    top_breeds = breed_traits.sort_values(by='Total_Score', ascending=False).head(3)

    # show results as bot messages and images
    for idx, row in top_breeds.iterrows():
        breed_name = row['Breed']
        add_bot_message(f"🐾 {breed_name} — Score: {int(row['Total_Score'])}")
        with st.spinner("Fetching images..."):
            images = fetch_images_for_breed(breed_name, max_images=MAX_IMAGES_PER_BREED)
        if images:
            # show first image inline using st.image (not in chat bubble but immediately visible)
            st.image(images[0], use_column_width=True)
            video_path = create_slideshow_video_from_pil(images, fps=VIDEO_FPS, duration_per_image=VIDEO_FRAME_DURATION)
            if video_path and os.path.exists(video_path):
                st.video(video_path)
        else:
            variants = normalize_breed_variants(breed_name)
            guessed_folder = variants[0].replace(" ", "_")
            url = f"https://github.com/maartenvandenbroeck/Dog-Breeds-Dataset/tree/master/breeds/{guessed_folder}"
            add_bot_message(f"Images not found; view potential folder: {url}")
        desc_text = generate_breed_description_from_traits(row, traits, top_n=4)
        add_bot_message("Description: " + desc_text)

    # show a small chart below
    st.subheader("📊 Score Distribution of Top Breeds")
    fig, ax = plt.subplots()
    ax.pie(top_breeds['Total_Score'], labels=top_breeds['Breed'], autopct="%1.1f%%")
    ax.set_title("Recommendation Weighting")
    st.pyplot(fig)

    # final actions
    choice = st.radio("What would you like to do next?", ["No, thank you", "Retake the test", "Download results"]) 
    if choice == "Retake the test":
        st.session_state.started = True
        st.session_state.step = 0
        st.session_state.answers = {}
        st.session_state.skipped = set()
        st.session_state.messages = []
        # push first question again
        if traits:
            t0 = traits[0]
            f = trait_desc[trait_desc['Trait'].str.strip().str.lower() == t0.strip().lower()]
            if not f.empty:
                r = f.iloc[0]
                add_bot_message(f"**{t0}** — {r['Description']}\n\n**1 = {r['Trait_1']}**, **5 = {r['Trait_5']}**")
        st.experimental_rerun()
    elif choice == "No, thank you":
        add_bot_message(f"Thanks for using the Dog Breed Recommender, {user_name}! 🐾")
    elif choice == "Download results":
        csv = top_breeds[["Breed", "Total_Score"]].to_csv(index=False)
        st.download_button("Download top breeds CSV", csv, file_name="top_breeds.csv", mime="text/csv")

    render_messages()
