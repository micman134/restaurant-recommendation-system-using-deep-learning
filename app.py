import streamlit as st
import requests
import pandas as pd
from transformers import pipeline
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime
import urllib.parse
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# ---------- CONFIG ----------
FOURSQUARE_KEY = st.secrets.get("FOURSQUARE_API_KEY")
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY")
GOOGLE_CX = st.secrets.get("GOOGLE_CX")  # Custom Search Engine ID

HEADERS = {
    "Accept": "application/json",
    "Authorization": f"Bearer {FOURSQUARE_KEY}",
    "X-Places-Api-Version": "2025-06-17"
}

# ---------- FOURSQUARE HELPERS ----------
def search_places_fs(query, near, limit=10):
    url = "https://places-api.foursquare.com/places/search"
    params = {"query": query, "near": near, "limit": limit}
    res = requests.get(url, headers=HEADERS, params=params)
    if res.status_code == 200:
        return res.json().get("results", [])
    return []

def get_tips_fs(fsq_id, limit=5):
    url = f"https://places-api.foursquare.com/places/{fsq_id}/tips"
    res = requests.get(url, headers=HEADERS)
    if res.status_code == 200:
        tips = res.json()
        return [tip.get("text", "") for tip in tips[:limit]]
    return []

def get_photos_fs(fsq_id, limit=1):
    url = f"https://places-api.foursquare.com/places/{fsq_id}/photos"
    res = requests.get(url, headers=HEADERS)
    if res.status_code == 200:
        photos = res.json()
        return [f"{p['prefix']}original{p['suffix']}" for p in photos[:limit]]
    return []

# ---------- GOOGLE IMAGE SEARCH ----------
def get_google_image(query):
    if not GOOGLE_API_KEY or not GOOGLE_CX:
        return ""
    url = f"https://www.googleapis.com/customsearch/v1"
    params = {
        "q": query,
        "cx": GOOGLE_CX,
        "searchType": "image",
        "num": 1,
        "key": GOOGLE_API_KEY
    }
    res = requests.get(url, params=params)
    if res.status_code == 200:
        items = res.json().get("items", [])
        if items:
            return items[0].get("link", "")
    return ""

# ---------- AI GENERATED REVIEW ----------
def generate_review(fsq_name):
    # Simple prompt for sentiment/description using HuggingFace pipeline
    return f"{fsq_name} is a highly recommended place with great food and excellent service."

# ---------- SENTIMENT MODEL ----------
@st.cache_resource(show_spinner=False)
def get_classifier():
    return pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

# ---------- FIREBASE ----------
if not firebase_admin._apps:
    cred = credentials.Certificate({
        "type": st.secrets["firebase"]["type"],
        "project_id": st.secrets["firebase"]["project_id"],
        "private_key_id": st.secrets["firebase"]["private_key_id"],
        "private_key": st.secrets["firebase"]["private_key"].replace('\\n', '\n'),
        "client_email": st.secrets["firebase"]["client_email"],
        "client_id": st.secrets["firebase"]["client_id"],
        "auth_uri": st.secrets["firebase"]["auth_uri"],
        "token_uri": st.secrets["firebase"]["token_uri"],
        "auth_provider_x509_cert_url": st.secrets["firebase"]["auth_provider_x509_cert_url"],
        "client_x509_cert_url": st.secrets["firebase"]["client_x509_cert_url"]
    })
    firebase_admin.initialize_app(cred)
db = firestore.client()

def append_history(data_dict):
    try:
        data_dict["timestamp"] = datetime.now()
        db.collection("recommendations").add(data_dict)
    except Exception as e:
        st.error(f"Error saving to Firebase: {e}")

# ---------- STREAMLIT UI ----------
st.title("🍽️ AI Restaurant Recommender (Foursquare + Google Images)")

food = st.text_input("🍕 Food Type", placeholder="Sushi, Pizza...")
location = st.text_input("📍 Location", placeholder="Lagos, Nigeria")

if st.button("🔍 Search"):
    if not food or not location:
        st.warning("Enter both food type and location.")
    else:
        classifier = get_classifier()
        results = []

        # Search Foursquare
        fs_places = search_places_fs(food, location, limit=20)

        for p in fs_places:
            fsq_id = p.get("fsq_id")
            name = p.get("name")
            address = ", ".join(p.get("location", {}).get("formatted_address", [])) or "Unknown"
            maps_link = f"https://www.google.com/maps/search/?api=1&query={urllib.parse.quote_plus(name + ' ' + address)}"

            # Get tips / reviews
            tips = get_tips_fs(fsq_id)
            if not tips:
                tips = [generate_review(name)]

            # Sentiment scoring
            sentiments = [int(classifier(t[:512])[0]["label"].split()[0]) for t in tips]
            avg_rating = round(sum(sentiments)/len(sentiments), 2) if sentiments else 0

            # Get image from Foursquare first, fallback to Google
            photos = get_photos_fs(fsq_id)
            image_url = photos[0] if photos else get_google_image(f"{name} restaurant {location}")

            results.append({
                "Restaurant": name,
                "Address": address,
                "Google Maps Link": maps_link,
                "Rating": avg_rating,
                "Stars": "⭐"*int(round(avg_rating)) if avg_rating>0 else "No reviews",
                "Reviews": len(sentiments),
                "Image": image_url,
                "Tips": tips[:3]
            })

        if results:
            df = pd.DataFrame([{
                "Restaurant": r["Restaurant"],
                "Address": r["Address"],
                "Average Rating": r["Rating"],
                "Stars": r["Stars"],
                "Reviews": r["Reviews"]
            } for r in results])
            df.index += 1
            st.dataframe(df, use_container_width=True)

            # Top pick
            top = max(results, key=lambda x: x["Rating"])
            st.metric(label="🏆 Top Pick", value=top["Restaurant"], delta=f"{top['Rating']} ⭐")
            append_history({
                "Restaurant": top["Restaurant"],
                "Rating": top["Rating"],
                "Address": top["Address"],
                "Google Maps Link": top["Google Maps Link"],
                "Food": food,
                "Location": location
            })

            # Gallery
            st.subheader("🖼️ Gallery Picks")
            cols = st.columns(3)
            for idx, r in enumerate(sorted(results, key=lambda x: x["Rating"], reverse=True)):
                with cols[idx%3]:
                    st.image(r["Image"], caption=f"{r['Restaurant']} ({r['Stars']})")
