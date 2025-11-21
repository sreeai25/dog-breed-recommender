# UPDATED app.py — with dynamic image fetching, corrected name flow, PDF download with images, simplified results

import streamlit as st
st.set_page_config(page_title="Dog Breed Recommender", layout="centered")

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
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image as RLImage, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch

# =====================
# CONFIG
# =====================
DATA_DIR = "data"
BREED_CSV = os.path.join(DATA_DIR, "breed_traits.csv")
TRAIT_CSV = os.path.join(DATA_DIR, "trait_descriptions.csv")
GITHUB_REPO_RAW = "https://raw.githubusercontent.com/maartenvandenbroeck/Dog-Breeds-Dataset/master/breeds"

MAX_IMAGES_PER_BREED = 3

# =====================
# LOAD DATA
# =====================
@st.cache_data
def load_data():
    breed_traits = pd.read_csv(BREED_CSV)
    trait_desc = pd.read_csv(TRAIT_CSV)

    # Basic cleanup
    for df in (breed_traits, trait_desc):
        for c in df.select_dtypes(include=[object]).columns:
            df[c] = df[c].astype(str).str.replace(' ', ' ', regex=False).str.replace('Â', '', regex=False)

    return breed_traits, trait_desc

breed_traits, trait_desc = load_data()
ignore_traits = ["Coat Type", "Coat Length"]
traits = [t for t in trait_desc["Trait"].tolist() if t not in ignore_traits]

# Ensure breed trait columns exist
for t in traits:
    if t not in breed_traits.columns:
        breed_traits[t] = 0
    breed_traits[t] = pd.to_numeric(breed_traits[t], errors="coerce").fillna(0)

# =====================
# IMAGE HELPERS
# =====================
def normalize_variants(name):
    raw = name.strip().replace("Â", "")
    versions = [raw]
    versions.append(raw.replace(" ", "_"))
    versions.append(raw.lower().replace(" ", "_"))
    versions.append("_".join(w.capitalize() for w in raw.split()))
    return list(dict.fromkeys(versions))

@st.cache_data(show_spinner=False)
def fetch_images(breed, max_n=MAX_IMAGES_PER_BREED):
    images = []
    for variant in normalize_variants(breed):
        for idx in range(1, max_n+1):
            for ext in ["jpg", "jpeg", "png"]:
                url = f"{GITHUB_REPO_RAW}/{quote(variant)}/{idx}.{ext}"
                try:
                    r = requests.get(url, timeout=5)
                    if r.status_code == 200 and 'image' in r.headers.get('content-type',''):
                        img = PILImage.open(BytesIO(r.content)).convert("RGB")
                        images.append(img)
                        break
                except Exception:
                    pass
        if len(images) >= max_n:
            break
    return images[:max_n]

# =====================
# PDF GENERATION
# =====================
def generate_pdf(user_name, top_breeds):
    pdf_path = "/tmp/recommendations.pdf"
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph(f"Dog Breed Recommendations for {user_name}", styles['Title']))
    story.append(Spacer(1, 12))

    for idx, row in top_breeds.iterrows():
        breed = row['Breed']
        story.append(Paragraph(f"<b>{breed}</b> (Score: {int(row['Total_Score'])})", styles['Heading2']))
        story.append(Spacer(1, 6))

        imgs = fetch_images(breed)
        for img in imgs:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
            img.save(tmp.name)
            story.append(RLImage(tmp.name, width=3*inch, height=3*inch))
            story.append(Spacer(1, 12))

    doc.build(story)
    return pdf_path

# =====================
# STREAMLIT UI
# =====================
st.title("🐶 Dog Breed Recommender")
st.markdown("**Welcome to Dog breed recommender.**")
st.markdown("I am **D**, your friendly assistant to help you choose your furry friend.")

# session states
for key, default in {
    "started": False,
    "step": 0,
    "answers": {},
    "skipped": set(),
    "name": "",
    "results": None
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# NAME INPUT
name = st.text_input("What is your name?", value=st.session_state.name)

if name and name != st.session_state.name:
    st.session_state.name = name
    st.success(f"Thank you for choosing me today, {name}. Let's start. Please rate the following traits you prefer in your dog. Click Start once ready.")

if st.button("Start"):
    st.session_state.started = True
    st.session_state.step = 0
    st.session_state.answers = {}
    st.session_state.skipped = set()

if not st.session_state.started:
    st.stop()

user_name = st.session_state.name

# QUIZ LOOP
if st.session_state.step < len(traits):
    trait = traits[st.session_state.step]
    row = trait_desc[trait_desc['Trait'] == trait].iloc[0]

    st.subheader(trait)

    with st.expander("Trait description and rating scale"):
        st.write(row['Description'])
        st.write(f"1 = {row['Trait_1']} — 5 = {row['Trait_5']}")

    val = st.slider("Importance?", 1, 5, st.session_state.answers.get(trait, 3))

    cols = st.columns(2)
    if cols[0].button("Next"):
        st.session_state.answers[trait] = val
        st.session_state.step += 1
        st.experimental_rerun()
    if cols[1].button("Skip"):
        st.session_state.skipped.add(trait)
        st.session_state.step += 1
        st.experimental_rerun()

else:
    # DONE — CALCULATE RESULTS
    def score(row):
        s = 0
        for t, imp in st.session_state.answers.items():
            s += row[t] * imp
        return s

    breed_traits['Total_Score'] = breed_traits.apply(score, axis=1)
    top3 = breed_traits.sort_values("Total_Score", ascending=False).head(3)
    st.session_state.results = top3

    st.header("My top 3 recommendations are…")

    # SHOW ONLY NAME + IMAGES
    for idx, row in top3.iterrows():
        st.subheader(row['Breed'])
        imgs = fetch_images(row['Breed'])
        if imgs:
            for img in imgs:
                st.image(img, use_column_width=True)
        else:
            st.info("Images not available.")
        st.write("---")

    # PDF DOWNLOAD
    pdf_path = generate_pdf(user_name, top3)
    with open(pdf_path, "rb") as f:
        st.download_button("Download recommendations as PDF", f, file_name="recommendations.pdf")

    # FINAL ACTIONS
    choice = st.radio("What would you like to do next?", ["Exit", "Retake the quiz", "Explain results"])

    if st.button("Confirm"):
        if choice == "Exit":
            st.success(f"Thank you for using my assistance today. Good luck with your furry friend, {user_name}!")
            st.stop()
        elif choice == "Retake the quiz":
            st.session_state.started = True
            st.session_state.step = 0
            st.session_state.answers = {}
            st.experimental_rerun()
        elif choice == "Explain results":
            st.subheader("Selection logic used")
            st.write("Calculates breed scores = sum(breed_trait_value * user_importance). The breed with the maximum score is recommended.")
