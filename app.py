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
from bs4 import BeautifulSoup

# Set page configuration
st.set_page_config(page_title="🍽️ Restaurant Recommender", layout="wide")

# Background and CSS (same as your original)
st.markdown("""
<style>
.stApp { background-image: url("https://images.unsplash.com/photo-1517248135467-4c7edcad34c4"); background-size: cover; background-position: center; background-repeat: no-repeat; background-attachment: fixed; }
.stApp:before { content: ""; position: absolute; top: 0; left: 0; right: 0; bottom: 0; background: rgba(0, 0, 0, 0.85); z-index: 0; }
#MainMenu, footer, header {visibility: hidden;}
.custom-footer {text-align: center; font-size: 14px; margin-top: 50px; padding: 20px; color: #aaa;}
.gallery-img-container {width: 100%; height: 250px; overflow: hidden; border-radius: 10px; margin-bottom: 10px;}
.gallery-img {width: 100%; height: 100%; object-fit: cover;}
.gallery-caption {text-align: center; margin-top: 5px;}
.map-link {color: #4CAF50 !important; text-decoration: none; font-weight: bold;}
.map-link:hover {text-decoration: underline;}
.stTabs [data-baseweb="tab-list"] {gap: 10px;}
.stTabs [data-baseweb="tab"] {padding: 8px 16px; border-radius: 8px 8px 0 0;}
.stTabs [aria-selected="true"] {background-color: #4CAF50; color: white;}
</style>
""", unsafe_allow_html=True)

# Autofocus
st.markdown("""
<script>
const foodInput = window.parent.document.querySelectorAll('input[type="text"]')[0];
if (foodInput) { foodInput.focus(); }
</script>
""", unsafe_allow_html=True)

# Load sentiment classifier
@st.cache_resource(show_spinner=False)
def get_classifier():
    return pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

# Initialize Firebase
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

def read_history():
    try:
        docs = db.collection("recommendations").stream()
        history_data = []
        for doc in docs:
            data = doc.to_dict()
            data["id"] = doc.id
            history_data.append(data)
        return history_data
    except Exception as e:
        st.error(f"Error reading from Firebase: {e}")
        return []

def append_history(data_dict):
    food = data_dict.get("Food", "").strip()
    location = data_dict.get("Location", "").strip()
    if not food or not location:
        return
    try:
        docs = db.collection("recommendations") \
                 .where(field_path="Restaurant", op_string="==", value=data_dict.get("Restaurant")) \
                 .where(field_path="Food", op_string="==", value=food) \
                 .where(field_path="Location", op_string="==", value=location) \
                 .stream()
        if len(list(docs)) > 0:
            return
        data_dict["timestamp"] = datetime.now()
        db.collection("recommendations").add(data_dict)
    except Exception as e:
        st.error(f"Error saving to Firebase: {e}")

# Session state initialization
if "page" not in st.session_state:
    st.session_state.page = "Recommend"

# Sidebar
with st.sidebar:
    st.markdown("## 🍽️ Menu")
    if st.button("Recommend"): st.session_state.page = "Recommend"
    if st.button("Deep Learning"): st.session_state.page = "Deep Learning"
    if st.button("History"): st.session_state.page = "History"
    if st.button("About"): st.session_state.page = "About"

# -------- PAGE: Recommend --------
if st.session_state.page == "Recommend":
    st.title("🍽️ AI Restaurant Recommender (Web Scraping)")
    st.markdown("Find top-rated restaurants near you using **Foursquare web scraping** and **AI sentiment analysis**.")

    col1, _ = st.columns([1,1])
    with col1:
        food = st.text_input("🍕 Food Type", placeholder="e.g., Sushi, Jollof, Pizza")
    col2, _ = st.columns([1,1])
    with col2:
        location = st.text_input("📍 Location", placeholder="e.g., Lagos, Nigeria")

    if st.button("🔍 Search"):
        if not food or not location:
            st.warning("⚠️ Enter both food type and location.")
        else:
            st.session_state.results = None
            st.session_state.df = None

            with st.spinner("Scraping Foursquare..."):
                # Build search URL
                query = urllib.parse.quote_plus(f"{food} in {location}")
                url = f"https://foursquare.com/explore?mode=url&near={location}&q={food}"

                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
                }

                response = requests.get(url, headers=headers)
                if response.status_code != 200:
                    st.error("❌ Failed to fetch data. Try again later.")
                else:
                    soup = BeautifulSoup(response.text, "html.parser")
                    results = []

                    # Scrape restaurant names and addresses
                    for card in soup.find_all("div", {"class": "venueCard"}):  # adjust selector if needed
                        name_tag = card.find("h4")
                        addr_tag = card.find("address")
                        if name_tag and addr_tag:
                            name = name_tag.text.strip()
                            address = addr_tag.text.strip()
                            maps_link = f"https://www.google.com/maps/search/{urllib.parse.quote_plus(name + ' ' + address)}"
                            results.append({
                                "Restaurant": name,
                                "Address": address,
                                "Google Maps Link": maps_link,
                                "Rating": 0,  # default, will analyze reviews later
                                "Stars": "No reviews",
                                "Reviews": 0,
                                "Tips": ["No reviews available"],
                                "Image": ""
                            })

                    if results:
                        st.session_state.results = results
                        df = pd.DataFrame([{
                            "Restaurant": r["Restaurant"],
                            "Address": r["Address"],
                            "Stars": r["Stars"],
                            "Reviews": r["Reviews"]
                        } for r in results])
                        df.index += 1
                        st.session_state.df = df
                        st.dataframe(df, use_container_width=True)
                    else:
                        st.error("❌ No restaurants found. Try different search terms.")

# The rest of your pages (Deep Learning, History, About) remain the same

# Footer
st.markdown('<div class="custom-footer">© 2025 AI Restaurant Recommender</div>', unsafe_allow_html=True)
