# app.py — Streamlit Dog Breed Recommender

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
    for df in (breed_traits, trait_desc):
        for c in df.select_dtypes(include=[object]).columns:
            df[c] = df[c].astype(str).str.replace(' ', ' ', regex=False).str.replace('Â', '', regex=False)
    return breed_traits, trait_desc

breed_traits, trait_desc = load_data()
ignore_traits = ["Coat Type", "Coat Length"]
traits = [t for t in trait_desc["Trait"].tolist() if t not in ignore_traits]

# Ensure numeric columns exist
for t in traits:
    if t not in breed_traits.columns:
        breed_traits[t] = 0
    breed_traits[t] = pd.to_numeric(breed_traits[t], errors="coerce").fillna(0)

# =====================
# IMAGE HELPERS
# =====================
def normalize_breed_variants(name):
    """
    Convert CSV breed names to plausible GitHub folder names.
    Returns a list of variants to try.
    """
    if not isinstance(name, str):
        name = str(name)
    
    name = name.replace("\xa0", " ").replace("Â", "").replace("â€™", "'").strip()

    # Handle parentheses: move content inside parentheses first
    paren_match = re.search(r"\((.*?)\)", name)
    if paren_match:
        inside = paren_match.group(1).strip()
        outside = re.sub(r"\(.*?\)", "", name).strip()
        main_name = f"{inside} {outside}"
    else:
        main_name = name

    main_name = re.sub(r"[^a-zA-Z ]", "", main_name).lower()
    words = [w[:-1] if w.endswith('s') else w for w in main_name.split()]
    words.append("dog")
    folder_name = " ".join(words).strip()
    folder_url = quote(folder_name)
    return [folder_name, folder_url, "_".join(words), "".join(words)]

@st.cache_data(show_spinner=False)
def fetch_images_for_breed(breed_name, max_images=MAX_IMAGES_PER_BREED):
    images = []
    variants = normalize_breed_variants(breed_name)
    exts = ["jpg", "jpeg", "png"]

    for folder in variants:
        for idx in range(1, max_images + 1):
            for ext in exts:
                url = f"{GITHUB_REPO_RAW}/{quote(folder)}/Image_{idx}.{ext}"
                try:
                    resp = requests.get(url, timeout=5)
                    if resp.status_code == 200 and "image" in resp.headers.get("content-type", ""):
                        img = PILImage.open(BytesIO(resp.content)).convert("RGB")
                        images.append((img, url))
                        break
                except:
                    continue
        if len(images) >= max_images:
            break

    # deduplicate images by size+mode
    uniq = []
    seen = set()
    for img, url in images:
        key = (img.size, img.mode)
        if key not in seen:
            uniq.append((img, url))
            seen.add(key)

    return uniq[:max_images]

def display_images_with_fallback(breed_name):
    with st.spinner(f"Fetching images for {breed_name}…"):
        images = fetch_images_for_breed(breed_name)
    if images:
        for img, url in images:
            st.image(img, use_column_width=True)
            st.markdown(f"[View image]({url})")
    else:
        folder = normalize_breed_variants(breed_name)[0].replace(" ", "_")
        url = f"https://github.com/maartenvandenbroeck/Dog-Breeds-Dataset/tree/master/breeds/{folder}"
        st.info(f"No images found for **{breed_name}**. You can view images here:")
        st.markdown(f"[Open GitHub folder for {breed_name}]({url})", unsafe_allow_html=True)

# =====================
# PDF GENERATION
# =====================
def generate_pdf(user_name, top_breeds):
    pdf_path = os.path.join(tempfile.gettempdir(), "recommendations.pdf")
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    story.append(Paragraph(f"Dog Breed Recommendations for {user_name}", styles['Title']))
    story.append(Spacer(1,12))

    for idx, row in top_breeds.iterrows():
        breed = row['Breed']
        story.append(Paragraph(f"{breed} (Score: {int(row['Total_Score'])})", styles['Heading2']))
        story.append(Spacer(1,6))
        imgs = fetch_images_for_breed(breed)
        for img, url in imgs:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
            img.save(tmp.name)
            story.append(RLImage(tmp.name, width=3*inch, height=3*inch))
            story.append(Spacer(1,6))
            story.append(Paragraph(f"<a href='{url}'>View image online</a>", styles['Normal']))
            story.append(Spacer(1,12))

    doc.build(story)
    return pdf_path

# =====================
# STREAMLIT UI
# =====================
st.title("🐶 Dog Breed Recommender")
st.markdown("**Welcome to Dog breed recommender.**")
st.markdown("I am **D**, your friendly assistant to help you choose your furry friend.")
st.markdown("**What is your name?**")

# session state defaults
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

name = st.text_input("", value=st.session_state.name)

cols_start = st.columns([1,1])
with cols_start[0]:
    if st.button("Start") or (name and st.session_state.name != name):
        st.session_state.name = name
        st.session_state.started = True
        st.session_state.step = 0
        st.session_state.answers = {}
        st.session_state.skipped = set()
        st.success(f"Thank you {name} for choosing me today. Let's start. Please rate the traits you prefer in your dog based on your lifestyle. Click Next to begin the quiz.")
with cols_start[1]:
    if st.button("Exit"):
        user = st.session_state.name or "friend"
        st.success(f"Thank you for using my assistance today, {user}. Good luck with your furry friend!")
        st.stop()

if not st.session_state.started:
    st.stop()

user_name = st.session_state.name
total_traits = len(traits)

# ---------------------
# QUIZ LOOP
# ---------------------
if st.session_state.step < total_traits:
    trait = traits[st.session_state.step]
    trow = trait_desc[trait_desc['Trait']==trait].iloc[0]
    st.subheader(f"Trait {st.session_state.step+1} of {total_traits}: {trait}")
    progress = (st.session_state.step+1)/total_traits
    st.progress(progress)

    with st.expander("Description & rating scale"):
        st.write(trow['Description'])
        st.write(f"1 = {trow['Trait_1']} — 5 = {trow['Trait_5']}")

    val = st.slider("Importance?", 1, 5, st.session_state.answers.get(trait,3), key=f"slider_{trait}")

    col_back, col_skip, col_next = st.columns([1,1,1])
    if col_back.button("⬅ Back"):
        if st.session_state.step > 0:
            st.session_state.step -= 1
        st.experimental_rerun()
    if col_skip.button("Skip ❌"):
        st.session_state.skipped.add(trait)
        if trait in st.session_state.answers:
            del st.session_state.answers[trait]
        st.session_state.step +=1
        st.experimental_rerun()
    if col_next.button("Next ➡"):
        st.session_state.answers[trait] = val
        if trait in st.session_state.skipped:
            st.session_state.skipped.remove(trait)
        st.session_state.step +=1
        st.experimental_rerun()

# ---------------------
# RESULTS
# ---------------------
else:
    st.info("Calculating results… please be patient!")
    used_traits = [t for t in traits if t in st.session_state.answers]
    if not used_traits:
        st.warning("No traits rated. Please rate at least one trait.")
        st.stop()

    # User rating table
    st.subheader("Your trait importance ratings:")
    df_user = pd.DataFrame({"Trait": used_traits, "Your Rating":[st.session_state.answers[t] for t in used_traits]})
    st.dataframe(df_user)

    # Calculate scores
    def score_row(row):
        s = 0
        for t in used_traits:
            s += row[t]*st.session_state.answers[t]
        return s

    breed_traits['Total_Score'] = breed_traits.apply(score_row, axis=1)
    top3 = breed_traits.sort_values("Total_Score", ascending=False).head(3)
    st.session_state.results = top3

    st.header("🐾 My top 3 recommendations are:")
    for idx, row in top3.iterrows():
        st.subheader(f"{row['Breed']} — Score: {int(row['Total_Score'])}")
        display_images_with_fallback(row['Breed'])
        st.write("---")

    # PDF
    pdf_path = generate_pdf(user_name, top3)
    with open(pdf_path, "rb") as f:
        st.download_button("Download recommendations as PDF", f, file_name="recommendations.pdf")

    # Pie chart
    try:
        fig, ax = plt.subplots()
        ax.pie(top3['Total_Score'], labels=top3['Breed'], autopct='%1.1f%%')
        ax.set_title("Recommendation Weighting")
        st.pyplot(fig)
    except:
        pass

    # Final options
    choice = st.radio("Next action:", ["Exit", "Retake Quiz", "Explain Results"], index=0)
    if st.button("Confirm"):
        if choice=="Exit":
            st.success(f"Thank you for using my assistance today, {user_name}. Good luck with your furry friend!")
            st.stop()
        elif choice=="Retake Quiz":
            st.session_state.started = True
            st.session_state.step = 0
            st.session_state.answers = {}
            st.session_state.skipped = set()
            st.experimental_rerun()
        elif choice=="Explain Results":
            st.subheader("Selection Logic & Contributions")
            st.write("Calculates breed scores = sum(breed_trait_value * user_importance).")
            for idx, row in top3.iterrows():
                breed = row['Breed']
                st.write(f"### {breed}")
                contributions = []
                for t in used_traits:
                    val = row[t]
                    imp = st.session_state.answers[t]
                    contributions.append([t, int(val), int(imp), int(val*imp)])
                df_contrib = pd.DataFrame(contributions, columns=["Trait","Breed Value","Your Importance","Contribution"])
                st.dataframe(df_contrib)
                st.write(f"**Total Score: {int(row['Total_Score'])}**")
