# app.py — Dog Breed Recommender with robust image fetching, trait table, PDF download
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
from urllib.parse import quote
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image as RLImage, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
import matplotlib.pyplot as plt

# =====================
# CONFIG
# =====================
DATA_DIR = "data"
BREED_CSV = os.path.join(DATA_DIR, "breed_traits.csv")
TRAIT_CSV = os.path.join(DATA_DIR, "trait_descriptions.csv")
GITHUB_REPO_RAW = "https://raw.githubusercontent.com/maartenvandenbroeck/Dog-Breeds-Dataset/master/breeds"
MAX_IMAGES_PER_BREED = 3

# =====================
# UTILITY FUNCTIONS
# =====================
@st.cache_data
def load_data():
    breed_traits = pd.read_csv(BREED_CSV)
    trait_desc = pd.read_csv(TRAIT_CSV)
    # Basic cleanup
    for df in (breed_traits, trait_desc):
        for c in df.select_dtypes(include=[object]).columns:
            df[c] = df[c].astype(str).str.replace(' ',' ', regex=False).str.replace('Â','', regex=False)
    return breed_traits, trait_desc

breed_traits, trait_desc = load_data()
ignore_traits = ["Coat Type", "Coat Length"]
traits = [t for t in trait_desc["Trait"].tolist() if t not in ignore_traits]
for t in traits:
    if t not in breed_traits.columns:
        breed_traits[t] = 0
    breed_traits[t] = pd.to_numeric(breed_traits[t], errors="coerce").fillna(0)

# ---------------------
# Image fetching
# ---------------------
def normalize_breed_variants(breed_name):
    breed_name = breed_name.lower().strip()
    words = [w for w in re.sub(r"[^a-z ]","",breed_name).split() if w not in ("dog","the")]
    variants = [
        "_".join(words),
        "".join(words),
        "_".join([w.capitalize() for w in words]),
        "".join([w.capitalize() for w in words]),
        "-".join(words)
    ]
    return list(dict.fromkeys(variants))

@st.cache_data(show_spinner=False)
def fetch_images_for_breed(breed_name, max_images=MAX_IMAGES_PER_BREED):
    images = []
    variants = normalize_breed_variants(breed_name)
    for folder in variants:
        for idx in range(1, max_images+1):
            for ext in ["jpg","jpeg","png"]:
                url = f"{GITHUB_REPO_RAW}/{folder}/{idx}.{ext}"
                try:
                    r = requests.get(url, timeout=5)
                    if r.status_code==200 and 'image' in r.headers.get('content-type',''):
                        img = PILImage.open(BytesIO(r.content)).convert("RGB")
                        images.append((img,url))
                        break
                except:
                    continue
        if len(images) >= max_images:
            break
    return images[:max_images]

# ---------------------
# PDF Generation
# ---------------------
def generate_pdf(user_name, top_breeds):
    pdf_path = "/tmp/recommendations.pdf"
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    story.append(Paragraph(f"Dog Breed Recommendations for {user_name}", styles['Title']))
    story.append(Spacer(1,12))

    for _, row in top_breeds.iterrows():
        breed = row['Breed']
        story.append(Paragraph(f"<b>{breed}</b> (Score: {int(row['Total_Score'])})", styles['Heading2']))
        story.append(Spacer(1,6))
        imgs = fetch_images_for_breed(breed)
        for img,_ in imgs:
            tmp = tempfile.NamedTemporaryFile(delete=False,suffix=".jpg")
            img.save(tmp.name)
            story.append(RLImage(tmp.name,width=3*inch,height=3*inch))
            story.append(Spacer(1,12))
    doc.build(story)
    return pdf_path

# =====================
# STREAMLIT UI
# =====================
st.title("🐶 Dog Breed Recommender")
st.markdown("**Welcome to Dog breed recommender.**")
st.markdown("I am **D**, your friendly assistant to help you choose your furry friend.")
st.markdown("What is your name?")

# Session state
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

# Name input
name = st.text_input("",value=st.session_state.name)
if st.session_state.name=="" and name:
    st.session_state.name = name
    st.success(f"Thank you {name} for choosing me today. Let's start. Please rate the following traits you prefer in your dog based on your personality and lifestyle. Click Start once ready.")

cols_ctrl = st.columns([1,1])
with cols_ctrl[0]:
    if st.button("Start") or (st.session_state.name!="" and name==st.session_state.name):
        st.session_state.started=True
        st.session_state.step=0
        st.session_state.answers={}
        st.session_state.skipped=set()
        st.experimental_rerun()
with cols_ctrl[1]:
    if st.button("Exit"):
        user = st.session_state.name or "friend"
        st.success(f"Thank you, {user}! Have a great day 🐾")
        st.stop()

if not st.session_state.started:
    st.stop()

user_name = st.session_state.name

# ---------------------
# Quiz loop
# ---------------------
if st.session_state.step < len(traits):
    trait = traits[st.session_state.step]
    row = trait_desc[trait_desc['Trait']==trait].iloc[0]

    st.subheader(trait)
    with st.expander("Trait description and rating scale"):
        st.write(row['Description'])
        st.write(f"1 = {row['Trait_1']} — 5 = {row['Trait_5']}")

    val = st.slider("Importance?",1,5,st.session_state.answers.get(trait,3))

    cols = st.columns([1,1,1])
    if cols[0].button("⬅ Back"):
        if st.session_state.step>0:
            st.session_state.step-=1
        st.experimental_rerun()
    if cols[1].button("Skip ❌"):
        st.session_state.skipped.add(trait)
        if trait in st.session_state.answers: del st.session_state.answers[trait]
        st.session_state.step+=1
        st.experimental_rerun()
    if cols[2].button("Next ➡"):
        st.session_state.answers[trait]=val
        if trait in st.session_state.skipped: st.session_state.skipped.remove(trait)
        st.session_state.step+=1
        st.experimental_rerun()

# ---------------------
# Results
# ---------------------
else:
    st.info("Loading results, please be patient...")

    # Trait importance table
    trait_table = []
    for t,val in st.session_state.answers.items():
        desc = trait_desc.loc[trait_desc['Trait']==t,'Description'].values[0]
        trait_table.append({"Trait":t,"Your Importance":val,"Description":desc})
    st.markdown("### Your trait importance ratings")
    st.dataframe(pd.DataFrame(trait_table))

    # Score calculation
    def score(row):
        s=0
        for t,val in st.session_state.answers.items():
            s+=row[t]*val
        return s

    breed_traits['Total_Score'] = breed_traits.apply(score,axis=1)
    top3 = breed_traits.sort_values("Total_Score",ascending=False).head(3)
    st.session_state.results = top3

    # Pie chart
    fig,ax = plt.subplots()
    ax.pie(top3['Total_Score'],labels=top3['Breed'],autopct='%1.1f%%')
    ax.set_title("Recommendation weighting")
    st.pyplot(fig)

    # Show breeds + images
    st.header("Top 3 recommended breeds")
    for _,row in top3.iterrows():
        st.subheader(f"{row['Breed']} (Score: {int(row['Total_Score'])})")
        images = fetch_images_for_breed(row['Breed'])
        if images:
            for img,_ in images:
                st.image(img,use_column_width=True)
        else:
            # fallback
            folder = normalize_breed_variants(row['Breed'])[0]
            st.write(f"Images not found. View GitHub folder: [link](https://github.com/maartenvandenbroeck/Dog-Breeds-Dataset/tree/master/breeds/{folder})")
        st.write("---")

    # PDF download
    pdf_path = generate_pdf(user_name, top3)
    with open(pdf_path,"rb") as f:
        st.download_button("Download recommendations as PDF",f,file_name="recommendations.pdf")

    # Bottom actions
    choice = st.radio("What would you like to do next?",["Exit","Retake the quiz","Explain results"])
    if st.button("Confirm"):
        if choice=="Exit":
            st.success(f"Thank you {user_name}! Enjoy your new furry friend 🐶")
            st.stop()
        elif choice=="Retake the quiz":
            st.session_state.started=True
            st.session_state.step=0
            st.session_state.answers={}
            st.session_state.skipped=set()
            st.experimental_rerun()
        elif choice=="Explain results":
            st.header("Explanation of results")
            for _,row in top3.iterrows():
                st.subheader(row['Breed'])
                contribs=[]
                for t,val in st.session_state.answers.items():
                    breed_val=row[t]
                    contrib=breed_val*val
                    contribs.append({"Trait":t,"Breed Value":breed_val,"Your Importance":val,"Contribution":contrib})
                st.dataframe(pd.DataFrame(contribs))
