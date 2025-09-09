import streamlit as st
import requests
import pandas as pd
from transformers import pipeline
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime
import urllib.parse
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import plotly.express as px

# --------- PAGE CONFIG ---------
st.set_page_config(page_title="🍽️ Restaurant Recommender", layout="wide")

# --------- STYLING ---------
st.markdown("""
<style>
.stApp {
    background-image: url("https://images.unsplash.com/photo-1517248135467-4c7edcad34c4");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}
.stApp:before {
    content: "";
    position: absolute; top:0; left:0; right:0; bottom:0;
    background: rgba(0,0,0,0.85); z-index:0;
}
#MainMenu, footer, header {visibility: hidden;}
.custom-footer {
    text-align: center; font-size:14px; margin-top:50px; padding:20px; color:#aaa;
}
.gallery-img-container { width:100%; height:250px; overflow:hidden; border-radius:10px; margin-bottom:10px; }
.gallery-img { width:100%; height:100%; object-fit:cover; }
.gallery-caption { text-align:center; margin-top:5px; }
.map-link { color:#4CAF50 !important; text-decoration:none; font-weight:bold; }
.map-link:hover { text-decoration:underline; }
.stTabs [data-baseweb="tab-list"] { gap:10px; }
.stTabs [data-baseweb="tab"] { padding:8px 16px; border-radius:8px 8px 0 0; }
.stTabs [aria-selected="true"] { background-color:#4CAF50; color:white; }
</style>
""", unsafe_allow_html=True)

# --------- LOAD SENTIMENT MODEL ---------
@st.cache_resource(show_spinner=False)
def get_classifier():
    return pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

classifier = get_classifier()

# --------- FIREBASE INITIALIZATION ---------
if not firebase_admin._apps:
    cred = credentials.Certificate({
        "type": st.secrets["firebase"]["type"],
        "project_id": st.secrets["firebase"]["project_id"],
        "private_key_id": st.secrets["firebase"]["private_key_id"],
        "private_key": st.secrets["firebase"]["private_key"].replace('\\n','\n'),
        "client_email": st.secrets["firebase"]["client_email"],
        "client_id": st.secrets["firebase"]["client_id"],
        "auth_uri": st.secrets["firebase"]["auth_uri"],
        "token_uri": st.secrets["firebase"]["token_uri"],
        "auth_provider_x509_cert_url": st.secrets["firebase"]["auth_provider_x509_cert_url"],
        "client_x509_cert_url": st.secrets["firebase"]["client_x509_cert_url"]
    })
    firebase_admin.initialize_app(cred)

db = firestore.client()

# --------- FIRESTORE FUNCTIONS ---------
def read_history():
    try:
        docs = db.collection("recommendations").stream()
        return [{**doc.to_dict(), "id": doc.id} for doc in docs]
    except Exception as e:
        st.error(f"Error reading from Firebase: {e}")
        return []

def append_history(data_dict):
    food = data_dict.get("Food", "").strip()
    location = data_dict.get("Location", "").strip()
    if not food or not location: return

    try:
        # Avoid duplicates
        docs = db.collection("recommendations") \
                 .where("Restaurant", "==", data_dict.get("Restaurant")) \
                 .where("Food", "==", food) \
                 .where("Location", "==", location).stream()
        if len(list(docs)) > 0: return

        data_dict["timestamp"] = datetime.now()
        db.collection("recommendations").add(data_dict)
    except Exception as e:
        st.error(f"Error saving to Firebase: {e}")

# --------- SESSION STATE ---------
if "page" not in st.session_state:
    st.session_state.page = "Recommend"

# --------- SIDEBAR ---------
with st.sidebar:
    st.markdown("## 🍽️ Menu")
    if st.button("Recommend"): st.session_state.page = "Recommend"
    if st.button("Deep Learning"): st.session_state.page = "Deep Learning"
    if st.button("History"): st.session_state.page = "History"
    if st.button("About"): st.session_state.page = "About"

# --------- HELPER FUNCTIONS ---------
def fetch_google_image(query):
    google_key = st.secrets.get("GOOGLE_API_KEY", "")
    cx = st.secrets.get("GOOGLE_CX", "")
    if not google_key or not cx: return ""
    url = f"https://www.googleapis.com/customsearch/v1?q={urllib.parse.quote_plus(query)}&cx={cx}&searchType=image&key={google_key}&num=1"
    try:
        res = requests.get(url).json()
        items = res.get("items")
        if items: return items[0]["link"]
    except: pass
    return ""

def analyze_sentiments(reviews):
    results = []
    for tip in reviews:
        res = classifier(tip[:512])[0]
        stars = int(res["label"].split()[0])
        results.append(stars)
    return results

# --------- PAGE: RECOMMEND ---------
if st.session_state.page == "Recommend":
    st.title("🍽️ AI Restaurant Recommender")
    st.markdown("Find top restaurants using **Foursquare** + AI sentiment analysis.")

    food = st.text_input("🍕 Food Type", placeholder="e.g., Sushi, Jollof, Pizza")
    location = st.text_input("📍 Location", placeholder="e.g., Lagos, Nigeria")
    api_key = st.secrets.get("FOURSQUARE_API_KEY", "")

    if st.button("🔍 Search"):
        if not food or not location:
            st.warning("⚠️ Enter both food type and location.")
        elif not api_key:
            st.error("❌ Foursquare API key missing.")
        else:
            st.session_state.results = []
            with st.spinner("Searching..."):
                #headers = {"Authorization": api_key, "accept": "application/json"}
                SERVICE_KEY = "M5XCF0QROQ4Q4G5RTPVV2E3HMLBTNMKX1MNWIMDCB4NU1SQF"  # Replace with your service key
                HEADERS = {"Accept": "application/json","Authorization": f"Bearer {SERVICE_KEY}","X-Places-Api-Version": "2025-06-17"}
                params = {"query": food, "near": location, "limit": 20}
                res = requests.get("https://places-api.foursquare.com/places/search", headers=HEADERS, params=params)
                restaurants = res.json().get("results", [])

                if not restaurants:
                    st.error("❌ No restaurants found.")
                else:
                    results = []
                    for r in restaurants:
                        fsq_id = r['fsq_id']
                        name = r['name']
                        address = r['location'].get('formatted_address', 'Unknown')
                        maps_link = f"https://www.google.com/maps/search/?api=1&query={urllib.parse.quote_plus(name+', '+address)}"

                        # Fetch tips
                        try:
                            tips_res = requests.get(f"https://places-api.foursquare.com/places/{fsq_place_id}/tips", headers=headers)
                            tips_data = tips_res.json() if tips_res.status_code == 200 else []
                        except: tips_data = []
                        review_texts = [tip.get("text","") for tip in tips_data[:5]] if tips_data else []

                        sentiments = analyze_sentiments(review_texts) if review_texts else []

                        # Fetch photo (fallback: Google Images)
                        photo_url = ""
                        try:
                            photo_res = requests.get(f"https://places-api.foursquare.com/v3/places/{fsq_place_id}/photos", headers=headers)
                            photos = photo_res.json() if photo_res.status_code == 200 else []
                            if photos: photo_url = f"{photos[0]['prefix']}original{photos[0]['suffix']}"
                        except: pass
                        if not photo_url: photo_url = fetch_google_image(f"{name} {location}")

                        avg_rating = round(sum(sentiments)/len(sentiments),2) if sentiments else 0
                        results.append({
                            "Restaurant": name,
                            "Address": address,
                            "Google Maps Link": maps_link,
                            "Rating": avg_rating,
                            "Stars": "⭐"*int(round(avg_rating)) if avg_rating else "No reviews",
                            "Reviews": len(sentiments),
                            "Image": photo_url,
                            "Tips": review_texts[:2] if review_texts else ["No reviews available"]
                        })
                    st.session_state.results = results

# --------- PAGE: DEEP LEARNING ---------
elif st.session_state.page == "Deep Learning":
    st.title("🤖 Deep Learning Explained")
    st.markdown("""
    AI analyzes restaurant reviews using **BERT sentiment model**.
    Steps:
    - Fetch restaurants from Foursquare API.
    - Retrieve user reviews (tips).
    - Compute sentiment scores → AI rating.
    - Recommend top restaurants.
    """)

# --------- PAGE: HISTORY ---------
elif st.session_state.page == "History":
    st.title("📚 Recommendation History")
    history_data = read_history()
    if not history_data:
        st.info("No history yet.")
    else:
        df_hist = pd.DataFrame(history_data).drop(columns=['id','timestamp'], errors='ignore')
        if 'Google Maps Link' in df_hist.columns:
            df_hist['Map'] = df_hist['Google Maps Link'].apply(lambda x: f"[📍 View on Map]({x})")
        df_hist.index += 1
        st.dataframe(df_hist, use_container_width=True)

# --------- PAGE: ABOUT ---------
elif st.session_state.page == "About":
    st.title("ℹ️ About")
    st.markdown("""
    AI Restaurant Recommender uses:
    - Foursquare API (places & reviews)
    - Google Images for restaurant photos
    - Hugging Face BERT sentiment analysis
    - Firebase Firestore for history tracking
    - Google Maps for navigation
    """)

# --------- FOOTER ---------
st.markdown('<div class="custom-footer">© 2025 AI Restaurant Recommender</div>', unsafe_allow_html=True)
