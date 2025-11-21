# app.py — Dog Breed Recommender
import streamlit as st

# ------------------------------
# Streamlit page config — MUST be first Streamlit call
# ------------------------------
st.set_page_config(page_title="Dog Breed Recommender", layout="centered")

# ------------------------------
# Imports
# ------------------------------
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
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image as RLImage, Spacer, Table
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch

# ------------------------------
# CONFIG
# ------------------------------
DATA_DIR = "data"
BREED_CSV = os.path.join(DATA_DIR, "breed_traits.csv")
TRAIT_CSV = os.path.join(DATA_DIR, "trait_descriptions.csv")
GITHUB_REPO_RAW = "https://raw.githubusercontent.com/maartenvandenbroeck/Dog-Breeds-Dataset/master/breeds"
MAX_IMAGES_PER_BREED = 3

# ------------------------------
# LOAD DATA
# ------------------------------
@st.cache_data
def load_data():
    breed_traits = pd.read_csv(BREED_CSV)
    trait_desc = pd.read_csv(TRAIT_CSV)
    for df in (breed_traits, trait_desc):
        for c in df.select_dtypes(include=[object]).columns:
            df[c] = df[c].astype(str).str.replace(' ', ' ', regex=False).str.replace('Â', '', regex=False)
    return breed_traits, trait_desc

breed_traits, trait_desc = load_data()
ignore_traits = ["Coat Type", "Coat Length"]
traits = [t for t in trait_desc["Trait"].tolist() if t not in ignore_traits]
for t in traits:
    if t not in breed_traits.columns:
        breed_traits[t] = 0
    breed_traits[t] = pd.to_numeric(breed_traits[t], errors="coerce").fillna(0)

# ------------------------------
# IMAGE HELPERS
# ------------------------------
def normalize_variants(name):
    raw = name.strip().replace("Â", "")
    versions = [raw, raw.replace(" ", "_"), raw.lower().replace(" ", "_")]
    versions.append("_".join([w.capitalize() for w in raw.split()]))
    return list(dict.fromkeys(versions))

@st.cache_data(show_spinner=False)
def fetch_images(breed, max_n=MAX_IMAGES_PER_BREED):
    images = []
    for variant in normalize_variants(breed):
        for idx in range(1, max_n+1):
            for ext in ["jpg","jpeg","png"]:
                url = f"{GITHUB_REPO_RAW}/{quote(variant)}/{idx}.{ext}"
                try:
                    r = requests.get(url, timeout=5)
                    if r.status_code == 200 and "image" in r.headers.get("content-type",""):
                        img = PILImage.open(BytesIO(r.content)).convert("RGB")
                        images.append(img)
                        break
                except Exception:
                    pass
        if len(images) >= max_n:
            break
    return images[:max_n]

# ------------------------------
# PDF GENERATION
# ------------------------------
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

# ------------------------------
# SESSION STATE INIT
# ------------------------------
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

# ------------------------------
# UI: WELCOME + NAME INPUT
# ------------------------------
st.title("🐶 Dog Breed Recommender")
st.markdown("**Welcome to Dog breed recommender.**")
st.markdown("I am **D**, your friendly assistant to help you choose your furry friend.")

name = st.text_input("What is your name?", value=st.session_state.name, key="name_input")

if name and name != st.session_state.name:
    st.session_state.name = name
    st.success(f"Thank you for choosing me today, {name}. Let's start. Please rate the traits you prefer in your dog based on your personality and lifestyle. Click Start once ready.")

start_col, exit_col = st.columns(2)
with start_col:
    if st.button("Start") and name:
        st.session_state.started = True
        st.session_state.step = 0
        st.session_state.answers = {}
        st.session_state.skipped = set()
        st.experimental_rerun()
with exit_col:
    if st.button("Exit"):
        st.success(f"Thank you! Have a great day, {name or 'friend'} 🐾")
        st.stop()

if not st.session_state.started:
    st.stop()

user_name = st.session_state.name

# ------------------------------
# QUIZ LOOP
# ------------------------------
total_traits = len(traits)
if st.session_state.step < total_traits:
    trait = traits[st.session_state.step]
    row = trait_desc[trait_desc['Trait'] == trait].iloc[0]

    st.markdown(f"**Question {st.session_state.step+1} of {total_traits}**")
    st.progress((st.session_state.step)/total_traits)

    st.subheader(trait)
    with st.expander("Trait description and rating scale"):
        st.write(row['Description'])
        st.write(f"1 = {row['Trait_1']} — 5 = {row['Trait_5']}")

    val = st.slider("Importance?", 1, 5, st.session_state.answers.get(trait, 3), key=f"slider_{trait}")

    col_back, col_next = st.columns(2)
    with col_back:
        if st.button("⬅ Back"):
            st.session_state.step = max(0, st.session_state.step-1)
            st.experimental_rerun()
    with col_next:
        if st.button("Next ➡"):
            st.session_state.answers[trait] = val
            st.session_state.step += 1
            st.experimental_rerun()

# ------------------------------
# RESULTS
# ------------------------------
else:
    with st.spinner("Loading results, please be patient…"):
        # compute scores
        def score(row):
            s = 0
            for t, imp in st.session_state.answers.items():
                s += row[t] * imp
            return s

        breed_traits['Total_Score'] = breed_traits.apply(score, axis=1)
        top3 = breed_traits.sort_values("Total_Score", ascending=False).head(3)
        st.session_state.results = top3

    st.header("My top 3 recommendations are…")
    # table of user scores
    st.subheader("Your trait ratings")
    df_answers = pd.DataFrame(list(st.session_state.answers.items()), columns=["Trait","Your Score"])
    st.table(df_answers)

    # show top breeds + images + scores
    for idx, row in top3.iterrows():
        st.subheader(f"{row['Breed']} — Score: {int(row['Total_Score'])}")
        imgs = fetch_images(row['Breed'])
        if imgs:
            for img in imgs:
                st.image(img, use_column_width=True)
        else:
            st.info("Images not available.")
        st.write("---")

    # pie chart
    fig, ax = plt.subplots()
    ax.pie(top3['Total_Score'], labels=top3['Breed'], autopct="%1.1f%%")
    ax.set_title("Recommendation Weighting")
    st.pyplot(fig)

    # PDF download
    pdf_path = generate_pdf(user_name, top3)
    with open(pdf_path, "rb") as f:
        st.download_button("Download recommendations as PDF", f, file_name="recommendations.pdf")

    # exit / retake options
    choice_col1, choice_col2 = st.columns(2)
    with choice_col1:
        if st.button("Retake Quiz"):
            st.session_state.started = True
            st.session_state.step = 0
            st.session_state.answers = {}
            st.experimental_rerun()
    with choice_col2:
        if st.button("Exit"):
            st.success(f"Thank you for using my assistance today. Good luck with your furry friend, {user_name}!")
            st.stop()
