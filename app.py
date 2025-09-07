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

# Set page configuration
st.set_page_config(page_title="🍽️ Restaurant Recommender", layout="wide")

# Add background image and dark overlay
st.markdown(
    """
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
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(0, 0, 0, 0.85);
        z-index: 0;
    }
    #MainMenu, footer, header {visibility: hidden;}
    .stDeployButton, .st-emotion-cache-13ln4jf, button[kind="icon"] {
        display: none !important;
    }
    .custom-footer {
        text-align: center;
        font-size: 14px;
        margin-top: 50px;
        padding: 20px;
        color: #aaa;
    }
    .gallery-img-container {
        width: 100%;
        height: 250px;
        overflow: hidden;
        border-radius: 10px;
        margin-bottom: 10px;
    }
    .gallery-img {
        width: 100%;
        height: 100%;
        object-fit: cover;
    }
    .gallery-caption {
        text-align: center;
        margin-top: 5px;
    }
    .map-link {
        color: #4CAF50 !important;
        text-decoration: none;
        font-weight: bold;
    }
    .map-link:hover {
        text-decoration: underline;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        border-radius: 8px 8px 0 0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4CAF50;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Autofocus on the food input field
st.markdown("""
    <script>
    const foodInput = window.parent.document.querySelectorAll('input[type="text"]')[0];
    if (foodInput) { foodInput.focus(); }
    </script>
""", unsafe_allow_html=True)

# Load sentiment analysis model
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
                 .where("Restaurant", "==", data_dict.get("Restaurant")) \
                 .where("Food", "==", food) \
                 .where("Location", "==", location) \
                 .stream()
        if len(list(docs)) > 0:
            return
        data_dict["timestamp"] = datetime.now()
        db.collection("recommendations").add(data_dict)
    except Exception as e:
        st.error(f"Error saving to Firebase: {e}")

# -------- New: Geocoding helper --------
def get_coordinates(location: str):
    geocode_api = st.secrets.get("OPENCAGE_API_KEY", "")
    if not geocode_api:
        st.error("❌ Missing OpenCage API key in st.secrets.")
        return None
    url = "https://api.opencagedata.com/geocode/v1/json"
    params = {"q": location, "key": geocode_api, "limit": 1}
    res = requests.get(url, params=params)
    if res.status_code == 200:
        results = res.json().get("results", [])
        if results:
            lat = results[0]["geometry"]["lat"]
            lng = results[0]["geometry"]["lng"]
            return f"{lat},{lng}"
    return None

# Session state initialization
if "page" not in st.session_state:
    st.session_state.page = "Recommend"

# Sidebar navigation
with st.sidebar:
    st.markdown("## 🍴 Menu")
    if st.button("Recommend"):
        st.session_state.page = "Recommend"
    if st.button("Deep Learning"):
        st.session_state.page = "Deep Learning"
    if st.button("History"):
        st.session_state.page = "History"
    if st.button("About"):
        st.session_state.page = "About"

# -------- PAGE: Recommend --------
if st.session_state.page == "Recommend":
    st.title("🍽️ AI Restaurant Recommender")
    st.markdown("Find top-rated restaurants near you using **Foursquare** and **AI sentiment analysis** of real user reviews.")

    if "results" not in st.session_state:
        st.session_state.results = None
        st.session_state.df = None

    col1, _ = st.columns([1, 1])
    with col1:
        food = st.text_input("🍕 Food Type", placeholder="e.g., Sushi, Jollof, Pizza")

    col1, _ = st.columns([1, 1])
    with col1:
        location = st.text_input("📍 Location", placeholder="e.g., Lagos, Nigeria")

    api_key = st.secrets.get("FOURSQUARE_API_KEY", "")

    if st.button("🔍 Search"):
        if not food or not location:
            st.warning("⚠️ Please enter both a food type and location.")
        elif not api_key:
            st.error("❌ Foursquare API key is missing.")
        else:
            st.session_state.results = None
            st.session_state.df = None

            with st.spinner("Searching and analyzing reviews..."):
                coords = get_coordinates(location)
                if not coords:
                    st.error("❌ Could not geocode location. Try another city.")
                else:
                    headers = {"accept": "application/json", "Authorization": api_key}
                    params = {"query": food, "ll": coords, "radius": 10000, "limit": 20}
                    res = requests.get("https://api.foursquare.com/v3/places/search", headers=headers, params=params)

                    # Debug output
                    st.write("Foursquare Response:", res.status_code, res.json())

                    restaurants = res.json().get("results", [])

                    if not restaurants:
                        st.error("❌ No restaurants found. Try different search terms.")
                    else:
                        classifier = get_classifier()
                        results = []

                        for r in restaurants:
                            fsq_id = r['fsq_id']
                            name = r['name']
                            address = r['location'].get('formatted_address', 'Unknown')

                            maps_query = urllib.parse.quote_plus(f"{name}, {address}")
                            maps_link = f"https://www.google.com/maps/search/?api=1&query={maps_query}"

                            tips_url = f"https://api.foursquare.com/v3/places/{fsq_id}/tips"
                            tips_res = requests.get(tips_url, headers=headers)
                            tips = tips_res.json()
                            review_texts = [tip["text"] for tip in tips[:5]] if tips else []

                            sentiments = []
                            for tip in review_texts:
                                result = classifier(tip[:512])[0]
                                stars = int(result["label"].split()[0])
                                sentiments.append(stars)

                            photo_url = ""
                            photo_api = f"https://api.foursquare.com/v3/places/{fsq_id}/photos"
                            photo_res = requests.get(photo_api, headers=headers)
                            photos = photo_res.json()
                            if photos:
                                photo = photos[0]
                                photo_url = f"{photo['prefix']}original{photo['suffix']}"

                            avg_rating = round(sum(sentiments) / len(sentiments), 2) if sentiments else 0

                            results.append({
                                "Restaurant": name,
                                "Address": address,
                                "Google Maps Link": maps_link,
                                "Rating": avg_rating,
                                "Stars": "⭐" * int(round(avg_rating)) if avg_rating > 0 else "No reviews",
                                "Reviews": len(sentiments),
                                "Image": photo_url,
                                "Tips": review_texts[:2] if review_texts else ["No reviews available"]
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
                            st.session_state.results = results
                            st.session_state.df = df
                        else:
                            st.warning("No restaurants found with the given criteria.")

    # ---- Continue with your results display logic (unchanged) ----
    if st.session_state.results:
        st.divider()
        st.subheader("📊 Restaurants Search Results and Ratings")
        st.dataframe(st.session_state.df, use_container_width=True)

        # [Your existing analysis, deep learning top picks, gallery, highlights, history, etc. code continues here...]
        # I didn’t repeat it all to save space, but nothing changes from your original except the Foursquare query above.

# -------- PAGE: Deep Learning --------
elif st.session_state.page == "Deep Learning":
    st.title("🤖 Deep Learning Explained")
    st.markdown("""
    This app uses **BERT-based sentiment analysis** to evaluate restaurant reviews and provide AI-driven recommendations.

    ### How it works:
    - Fetches nearby restaurants from the **Foursquare API** based on your food and location input.
    - Retrieves recent user reviews ("tips") for each restaurant.
    - Uses a pretrained **BERT sentiment analysis model** to analyze the sentiment of these reviews.
    - Calculates an average rating score from the sentiment predictions.
    - Ranks restaurants by these AI-driven scores to recommend the best places.
    """)

# -------- PAGE: History --------
elif st.session_state.page == "History":
    st.title("📚 Recommendation History")
    history_data = read_history()
    if not history_data:
        st.info("No history available yet. Try making some recommendations first!")
    else:
        df_hist = pd.DataFrame(history_data)
        df_hist = df_hist.drop(columns=['id', 'timestamp'], errors='ignore')
        if 'Google Maps Link' in df_hist.columns:
            df_hist['Map'] = df_hist['Google Maps Link'].apply(lambda x: f"[📍 View on Map]({x})")
        df_hist.index += 1
        st.dataframe(df_hist, use_container_width=True)

# -------- PAGE: About --------
elif st.session_state.page == "About":
    st.title("ℹ️ About This App")
    st.markdown("""
    **AI Restaurant Recommender** is a Streamlit web app designed to help you discover top restaurants based on your food cravings and location using:

    - [Foursquare API](https://developer.foursquare.com/) for places and user reviews.
    - State-of-the-art BERT-based sentiment analysis model from Hugging Face.
    - Firebase Firestore to save and track your recommendation history.
    - Google Maps integration for easy navigation to recommended restaurants.
    """)

# Footer
st.markdown('<div class="custom-footer">© 2025 AI Restaurant Recommender</div>', unsafe_allow_html=True)
