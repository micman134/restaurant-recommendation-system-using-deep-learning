import streamlit as st
import requests
import pandas as pd
from transformers import pipeline
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime
import urllib.parse

# Page config
st.set_page_config(page_title="🍽️ Restaurant Recommender", layout="wide")

# Background & overlay
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
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(0, 0, 0, 0.85);
        z-index: 0;
    }
    #MainMenu, footer, header {visibility: hidden;}
    .custom-footer {
        text-align: center;
        font-size: 14px;
        margin-top: 50px;
        padding: 20px;
        color: #aaa;
    }
    .gallery-img-container {
        width: 100%; height: 250px; overflow: hidden;
        border-radius: 10px; margin-bottom: 10px;
    }
    .gallery-img {
        width: 100%; height: 100%; object-fit: cover;
    }
    .gallery-caption {
        text-align: center; margin-top: 5px;
    }
    .map-link {
        color: #4CAF50 !important; text-decoration: none; font-weight: bold;
    }
    .map-link:hover {
        text-decoration: underline;
    }
    </style>
""", unsafe_allow_html=True)

# Autofocus food input
st.markdown("""
    <script>
    const foodInput = window.parent.document.querySelectorAll('input[type="text"]')[0];
    if (foodInput) { foodInput.focus(); }
    </script>
""", unsafe_allow_html=True)

# Load classifier
@st.cache_resource(show_spinner=False)
def get_classifier():
    return pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

# Firebase init
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
        return [doc.to_dict() for doc in docs]
    except Exception as e:
        st.error(f"Error reading Firebase: {e}")
        return []

def append_history(data):
    if not data.get("Food") or not data.get("Location"): return
    try:
        existing = db.collection("recommendations").where("Restaurant", "==", data["Restaurant"]) \
            .where("Food", "==", data["Food"]).where("Location", "==", data["Location"]).stream()
        if list(existing): return
        data["timestamp"] = datetime.now()
        db.collection("recommendations").add(data)
        st.success("Saved to history!")
    except Exception as e:
        st.error(f"Error saving to Firebase: {e}")

# Session state
if "page" not in st.session_state:
    st.session_state.page = "Recommend"

# Sidebar nav
with st.sidebar:
    st.markdown("## 🍽️ Menu")
    if st.button("Recommend"): st.session_state.page = "Recommend"
    if st.button("Deep Learning"): st.session_state.page = "Deep Learning"
    if st.button("History"): st.session_state.page = "History"
    if st.button("About"): st.session_state.page = "About"

# --- RECOMMEND PAGE ---
if st.session_state.page == "Recommend":
    st.title("🍽️ AI Restaurant Recommender")
    st.markdown("Discover top-rated restaurants using Foursquare + AI-powered review analysis.")

    if "results" not in st.session_state:
        st.session_state.results = None
        st.session_state.df = None

    # Food and location inputs
    col1, _ = st.columns([1, 1])
    with col1:
        food = st.text_input("🍕 Food Type", placeholder="e.g., Pizza, Sushi, Jollof", key="food_input")

    col1, col2 = st.columns([3, 1])
    with col1:
        location = st.text_input("📍 Location", placeholder="e.g., Lagos, Nigeria", key="location_input")
    with col2:
        if st.button("📍 Detect My Location"):
            try:
                res = requests.get("https://ipinfo.io/json")
                if res.status_code == 200:
                    data = res.json()
                    city = data.get("city", "")
                    country = data.get("country", "")
                    auto_location = f"{city}, {country}".strip(', ')
                    st.session_state.location_input = auto_location
                    st.success(f"Detected: {auto_location}")
                else:
                    st.warning("Couldn't detect location.")
            except Exception as e:
                st.error(f"Location error: {e}")

    api_key = st.secrets.get("FOURSQUARE_API_KEY", "")
    if st.button("🔍 Search"):
        if not food or not location:
            st.warning("Please enter food and location.")
        elif not api_key:
            st.error("Missing Foursquare API key.")
        else:
            st.session_state.results = None
            st.session_state.df = None

            with st.spinner("Analyzing..."):
                headers = {"accept": "application/json", "Authorization": api_key}
                params = {"query": food, "near": location, "limit": 20}
                res = requests.get("https://api.foursquare.com/v3/places/search", headers=headers, params=params)
                restaurants = res.json().get("results", [])

                if not restaurants:
                    st.error("No restaurants found.")
                else:
                    classifier = get_classifier()
                    results = []

                    for r in restaurants:
                        fsq_id = r["fsq_id"]
                        name = r["name"]
                        address = r["location"].get("formatted_address", "Unknown")
                        map_link = f"https://www.google.com/maps/search/?api=1&query={urllib.parse.quote_plus(name + ', ' + address)}"

                        tips_url = f"https://api.foursquare.com/v3/places/{fsq_id}/tips"
                        tips = requests.get(tips_url, headers=headers).json()
                        review_texts = [tip["text"] for tip in tips[:5]] if tips else []

                        sentiments = []
                        for text in review_texts:
                            result = classifier(text[:512])[0]
                            stars = int(result["label"].split()[0])
                            sentiments.append(stars)

                        avg_rating = round(sum(sentiments) / len(sentiments), 2) if sentiments else 0

                        photo_url = ""
                        photo_api = f"https://api.foursquare.com/v3/places/{fsq_id}/photos"
                        photos = requests.get(photo_api, headers=headers).json()
                        if photos:
                            photo = photos[0]
                            photo_url = f"{photo['prefix']}original{photo['suffix']}"

                        results.append({
                            "Restaurant": name,
                            "Address": address,
                            "Google Maps Link": map_link,
                            "Rating": avg_rating,
                            "Stars": "⭐" * int(round(avg_rating)) if avg_rating else "No reviews",
                            "Reviews": len(sentiments),
                            "Image": photo_url,
                            "Tips": review_texts[:2] if review_texts else ["No reviews"]
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

    if st.session_state.results:
        st.divider()
        st.subheader("📊 Restaurants & Ratings")
        st.dataframe(st.session_state.df, use_container_width=True)

        reviewed = [r for r in st.session_state.results if r["Reviews"] > 0]
        top3 = sorted(reviewed, key=lambda x: x["Rating"], reverse=True)[:3]

        st.divider()
        st.subheader("🏅 Top AI Picks")
        medals = ["🥇 1st", "🥈 2nd", "🥉 3rd"]
        colors = ["#FFD700", "#C0C0C0", "#CD7F32"]
        cols = st.columns(3)

        for i, (col, medal, color) in enumerate(zip(cols, medals, colors)):
            if i < len(top3):
                r = top3[i]
                with col:
                    st.markdown(f"""
                        <div style="background-color: {color}; border-radius: 15px; padding: 20px; text-align: center;">
                            <div style="font-size: 22px;">{medal}</div>
                            <div style="font-size: 18px;">{r['Restaurant']}</div>
                            <div>{r['Stars']} ({r['Rating']})</div>
                            <a href="{r['Google Maps Link']}" target="_blank" class="map-link">📍 locate restaurant</a>
                        </div>
                    """, unsafe_allow_html=True)

        st.divider()
        st.subheader("🖼️ Gallery Pick")
        gallery = [r for r in st.session_state.results if r["Image"]]
        gallery_cols = st.columns(3)
        for idx, r in enumerate(gallery):
            with gallery_cols[idx % 3]:
                st.markdown(f"""
                    <div class="gallery-img-container">
                        <img src="{r['Image']}" class="gallery-img" />
                    </div>
                    <div class="gallery-caption">
                        <strong>{r['Restaurant']}</strong><br>
                        ⭐ {r['Rating']}<br>
                        <a href="{r['Google Maps Link']}" class="map-link" target="_blank">📍 View</a>
                    </div>
                """, unsafe_allow_html=True)

        if reviewed:
            top = max(reviewed, key=lambda x: x["Rating"])
            st.metric(label="🏆 Top Pick", value=top["Restaurant"], delta=f"{top['Rating']} ⭐")
            append_history({
                "Restaurant": top["Restaurant"],
                "Rating": top["Rating"],
                "Address": top["Address"],
                "Google Maps Link": top["Google Maps Link"],
                "Food": food,
                "Location": location
            })

# --- DEEP LEARNING ---
elif st.session_state.page == "Deep Learning":
    st.title("🤖 Deep Learning Explained")
    st.markdown("""
    - Uses BERT sentiment model to analyze user reviews
    - Gets top restaurants from Foursquare API
    - Combines review stars into intelligent restaurant rankings
    """)

# --- HISTORY ---
elif st.session_state.page == "History":
    st.title("📚 History")
    data = read_history()
    if not data:
        st.info("No history yet.")
    else:
        df = pd.DataFrame(data)
        df = df.drop(columns=['timestamp'], errors='ignore')
        if 'Google Maps Link' in df:
            df['Map'] = df['Google Maps Link'].apply(lambda x: f"[📍 View]({x})")
        st.dataframe(df, use_container_width=True)

# --- ABOUT ---
elif st.session_state.page == "About":
    st.title("ℹ️ About This App")
    st.markdown("""
    - Built with Streamlit
    - Uses Foursquare for restaurants and reviews
    - Sentiment analysis powered by HuggingFace Transformers
    - Firebase stores recommendation history
    """)

# Footer
st.markdown('<div class="custom-footer">© 2025 AI Restaurant Recommender</div>', unsafe_allow_html=True)
