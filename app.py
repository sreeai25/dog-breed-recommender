# app.py
import streamlit as st
st.set_page_config(page_title="Dog Breed Recommender", layout="centered")

import pandas as pd
import numpy as np
import requests
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

# ------------------
# CONFIG
# ------------------
DATA_DIR = "data"
BREED_CSV = os.path.join(DATA_DIR, "breed_traits.csv")
TRAIT_CSV = os.path.join(DATA_DIR, "trait_descriptions.csv")
GITHUB_RAW = "https://raw.githubusercontent.com/maartenvandenbroeck/Dog-Breeds-Dataset/master/breeds"

MAX_IMAGES = 3

# ------------------
# LOAD DATA
# ------------------
@st.cache_data
def load_data():
    breed_traits = pd.read_csv(BREED_CSV)
    trait_desc = pd.read_csv(TRAIT_CSV)
    # clean
    for df in (breed_traits, trait_desc):
        for c in df.select_dtypes(include=[object]).columns:
            df[c] = df[c].astype(str).str.replace(' ',' ', regex=False).str.replace('Â','', regex=False)
    return breed_traits, trait_desc

breed_traits, trait_desc = load_data()
ignore_traits = ["Coat Type", "Coat Length"]
traits = [t for t in trait_desc["Trait"].tolist() if t not in ignore_traits]

# ensure numeric columns
for t in traits:
    if t not in breed_traits.columns:
        breed_traits[t] = 0
    breed_traits[t] = pd.to_numeric(breed_traits[t], errors="coerce").fillna(0)

# ------------------
# IMAGE HELPERS
# ------------------
def normalize_variants(name):
    raw = name.strip()
    variants = [raw, raw.replace(" ","_"), raw.lower().replace(" ","_"), "_".join([w.capitalize() for w in raw.split()])]
    return list(dict.fromkeys(variants))

@st.cache_data(show_spinner=False)
def fetch_images(breed, max_n=MAX_IMAGES):
    images = []
    for variant in normalize_variants(breed):
        for i in range(1,max_n+1):
            for ext in ["jpg","jpeg","png"]:
                url = f"{GITHUB_RAW}/{quote(variant)}/{i}.{ext}"
                try:
                    r = requests.get(url, timeout=5)
                    if r.status_code==200 and 'image' in r.headers.get('content-type',''):
                        img = PILImage.open(BytesIO(r.content)).convert("RGB")
                        images.append((img,url))
                        break
                except: pass
        if len(images)>=max_n:
            break
    return images[:max_n]

# ------------------
# PDF GENERATION
# ------------------
def generate_pdf(user_name, top3):
    pdf_path = "/tmp/dog_recommendations.pdf"
    doc = SimpleDocTemplate(pdf_path,pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    story.append(Paragraph(f"Dog Breed Recommendations for {user_name}",styles['Title']))
    story.append(Spacer(1,12))

    for idx,row in top3.iterrows():
        breed = row['Breed']
        story.append(Paragraph(f"<b>{breed}</b> (Score: {int(row['Total_Score'])})",styles['Heading2']))
        story.append(Spacer(1,6))
        imgs = fetch_images(breed)
        for img,_ in imgs:
            tmp = tempfile.NamedTemporaryFile(delete=False,suffix=".jpg")
            img.save(tmp.name)
            story.append(RLImage(tmp.name,width=3*inch,height=3*inch))
            story.append(Spacer(1,12))
    doc.build(story)
    return pdf_path

# ------------------
# STREAMLIT UI
# ------------------
st.title("🐶 Dog Breed Recommender")
st.markdown("**Welcome!** I am **D**, your friendly assistant to help you choose your furry friend.")

# SESSION STATE
for k,v in {"started":False,"step":0,"answers":{},"name":"","results":None}.items():
    if k not in st.session_state:
        st.session_state[k]=v

# NAME INPUT + START/EXIT/RETAKE
cols = st.columns([1,1,1])
name = st.text_input("Enter your name:",value=st.session_state.name)
with cols[0]:
    if st.button("Start"):
        if not name: st.warning("Enter your name first.")
        else:
            st.session_state.started=True
            st.session_state.name=name
            st.session_state.step=0
            st.session_state.answers={}
with cols[1]:
    if st.button("Exit"):
        st.success("Thank you! Come back any time 🐾")
        st.stop()
with cols[2]:
    if st.button("Retake Quiz"):
        st.session_state.started=True
        st.session_state.step=0
        st.session_state.answers={}
        st.experimental_rerun()

if not st.session_state.started: st.stop()
user_name = st.session_state.name
st.success(f"Thank you for choosing me today, {user_name}!")

# ------------------
# QUIZ LOOP
# ------------------
total = len(traits)
if st.session_state.step < total:
    trait = traits[st.session_state.step]
    trow = trait_desc[trait_desc['Trait']==trait].iloc[0]
    st.subheader(f"{trait} ({st.session_state.step+1}/{total})")
    st.progress((st.session_state.step+1)/total)

    with st.expander("Description and scale"):
        st.write(trow['Description'])
        st.write(f"1 = {trow['Trait_1']} — 5 = {trow['Trait_5']}")

    val = st.slider("Importance?",1,5,st.session_state.answers.get(trait,3))

    cols2 = st.columns([1,1])
    if cols2[0].button("⬅ Back"):
        if st.session_state.step>0: st.session_state.step-=1
        st.experimental_rerun()
    if cols2[1].button("Next ➡"):
        st.session_state.answers[trait]=val
        st.session_state.step+=1
        st.experimental_rerun()

# ------------------
# RESULTS
# ------------------
else:
    st.info("Loading results, please be patient…")

    def score(row):
        return sum(row[t]*imp for t,imp in st.session_state.answers.items())
    breed_traits['Total_Score'] = breed_traits.apply(score,axis=1)
    top3 = breed_traits.sort_values("Total_Score",ascending=False).head(3)
    st.session_state.results = top3

    st.header("Top 3 Recommendations")
    # user table
    st.subheader("Your Trait Importance")
    st.table(pd.DataFrame({"Trait":list(st.session_state.answers.keys()),
                           "Importance":list(st.session_state.answers.values())}))

    # display breeds with score + images + reference links
    scores=[]
    for _,row in top3.iterrows():
        st.subheader(f"{row['Breed']} — Score: {int(row['Total_Score'])}")
        imgs = fetch_images(row['Breed'])
        if imgs:
            for img,url in imgs:
                st.image(img,use_column_width=True)
                st.markdown(f"[Reference image link]({url})")
        else:
            st.info("Images not available.")
        st.write("---")
        scores.append(row['Total_Score'])

    # pie chart
    fig,ax = plt.subplots()
    ax.pie(scores,labels=top3['Breed'],autopct='%1.1f%%')
    ax.set_title("Recommendation Weighting")
    st.pyplot(fig)

    # PDF download
    pdf_path = generate_pdf(user_name,top3)
    with open(pdf_path,"rb") as f:
        st.download_button("Download recommendations as PDF (with images)",f,file_name="dog_recommendations.pdf")

    # final actions
    cols3 = st.columns([1,1,1])
    with cols3[0]:
        if st.button("Exit"):
            st.success(f"Thank you for using my assistance today. Good luck with your furry friend, {user_name}!")
            st.stop()
    with cols3[1]:
        if st.button("Retake Quiz"):
            st.session_state.started=True
            st.session_state.step=0
            st.session_state.answers={}
            st.experimental_rerun()
    with cols3[2]:
        if st.button("Explain Results"):
            st.subheader("Selection Logic")
            st.write("Calculates breed scores = sum(breed_trait_value * user_importance). The breed with max score is recommended.")
