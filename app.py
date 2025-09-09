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

# --------- PAGE CONFIGURATION ---------
st.set_page_config(page_title="🍽️ Restaurant Recommender", layout="wide")

# --------- CSS STYLING ---------
st.markdown("""
<style>
.stApp {
    background-image: url("https://images.unsplash.com/photo-1517248135467-4c7edcad34c4");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}
.stApp:before {
    content: "";
    position: absolute;
    top: 0; left: 0; right: 0; bottom: 0;
    background: rgba(0,0,0,0.85);
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
    width: 100%;
    height: 250px;
    overflow: hidden;
    border-radius: 10px;
    margin-bottom: 10px;
}
.gallery-img { width: 100%; height: 100%; object-fit: cover; }
.gallery-caption { text-align: center; margin-top: 5px; }
.map-link { color: #4CAF50 !important; text-decoration: none; font-weight: bold; }
.map-link:hover { text-decoration: underline; }
.stTabs [data-baseweb="tab-list"] { gap: 10px; }
.stTabs [data-baseweb="tab"] { padding: 8px 16px; border-radius: 8px 8px 0 0; }
.stTabs [aria-selected="true"] { background-color: #4CAF50; color: white; }
</style>
""", unsafe_allow_html=True)

# --------- SENTIMENT MODEL ---------
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

# --------- FIRESTORE FUNCTIONS ---------
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
        # Check duplicate
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

# --------- RECOMMEND PAGE ---------
if st.session_state.page == "Recommend":
    st.title("🍽️ AI Restaurant Recommender")
    st.markdown("Find top-rated restaurants using **Foursquare** and **AI sentiment analysis**.")

    if "results" not in st.session_state:
        st.session_state.results = None
        st.session_state.df = None

    food = st.text_input("🍕 Food Type", placeholder="e.g., Sushi, Jollof, Pizza")
    location = st.text_input("📍 Location", placeholder="e.g., Lagos, Nigeria")

    SERVICE_KEY = st.secrets.get("FOURSQUARE_SERVICE_KEY", "")
    if st.button("🔍 Search"):
        if not food or not location:
            st.warning("⚠️ Enter both food type and location.")
        elif not SERVICE_KEY:
            st.error("❌ Foursquare Service Key is missing.")
        else:
            st.session_state.results = None
            st.session_state.df = None
            with st.spinner("Searching and analyzing reviews..."):
                headers = {
                    "Accept": "application/json",
                    "Authorization": f"Bearer {SERVICE_KEY}",
                    "X-Places-Api-Version": "2025-06-17"
                }
                params = {"query": food, "near": location, "limit": 20}
                res = requests.get("https://places-api.foursquare.com/places/search", headers=headers, params=params)
                if res.status_code != 200:
                    st.error(f"❌ Foursquare API error: {res.status_code} {res.text}")
                else:
                    restaurants = res.json().get("results", [])
                    if not restaurants:
                        st.error("❌ No restaurants found. Try different search terms.")
                    else:
                        results = []
                        for r in restaurants:
                            fsq_id = r['fsq_place_id']
                            name = r['name']
                            address = r['location'].get('formatted_address', 'Unknown')
                            maps_query = urllib.parse.quote_plus(f"{name}, {address}")
                            maps_link = f"https://www.google.com/maps/search/?api=1&query={maps_query}"

                            # Tips
                            try:
                                tips_res = requests.get(f"https://places-api.foursquare.com/places/{fsq_id}/tips", headers=headers)
                                tips_data = tips_res.json() if tips_res.status_code == 200 else []
                            except:
                                tips_data = []
                            review_texts = [tip.get("text","") for tip in tips_data[:5]] if tips_data else []

                            # Sentiment
                            sentiments = [int(classifier(tip[:512])[0]["label"].split()[0]) for tip in review_texts]

                            # Photos
                            try:
                                photo_res = requests.get(f"https://places-api.foursquare.com/places/{fsq_id}/photos", headers=headers)
                                photos = photo_res.json() if photo_res.status_code == 200 else []
                            except:
                                photos = []
                            photo_url = f"{photos[0]['prefix']}original{photos[0]['suffix']}" if photos else ""

                            avg_rating = round(sum(sentiments)/len(sentiments),2) if sentiments else 0
                            results.append({
                                "Restaurant": name,
                                "Address": address,
                                "Google Maps Link": maps_link,
                                "Rating": avg_rating,
                                "Stars": "⭐"*int(round(avg_rating)) if avg_rating>0 else "No reviews",
                                "Reviews": len(sentiments),
                                "Image": photo_url,
                                "Tips": review_texts[:2] if review_texts else ["No reviews available"]
                            })

                        st.session_state.results = results
                        st.session_state.df = pd.DataFrame([{
                            "Restaurant": r["Restaurant"],
                            "Address": r["Address"],
                            "Average Rating": r["Rating"],
                            "Stars": r["Stars"],
                            "Reviews": r["Reviews"]
                        } for r in results])
                        st.session_state.df.index += 1

    # Display results
    if st.session_state.results:
        st.divider()
        st.subheader("📊 Restaurants Search Results")
        st.dataframe(st.session_state.df, use_container_width=True)

        # Top 3 AI Picks
        reviewed = [r for r in st.session_state.results if r["Reviews"]>0]
        top3 = sorted(reviewed, key=lambda x:x["Rating"], reverse=True)[:3] if reviewed else []

        st.divider()
        st.subheader("🏅 AI Top Picks")
        cols = st.columns(3)
        medals = ["🥇 1st","🥈 2nd","🥉 3rd"]
        colors = ["#FFD700","#C0C0C0","#CD7F32"]
        for i,(col,medal,color) in enumerate(zip(cols,medals,colors)):
            if i<len(top3):
                r = top3[i]
                with col:
                    st.markdown(f"""
                    <div style="background-color:{color}; border-radius:15px; padding:20px; text-align:center; color:black; font-weight:bold;">
                        <div style="font-size:22px; margin-bottom:10px;">{medal}</div>
                        <div style="font-size:18px; margin-bottom:8px;">{r['Restaurant']}</div>
                        <div style="font-size:15px; margin-bottom:8px;">{r['Address']}</div>
                        <div style="font-size:16px;">{r['Stars']} ({r['Rating']})</div>
                        <div style="margin-top:10px;">
                            <a href="{r['Google Maps Link']}" target="_blank" class="map-link">📍 locate restaurant</a>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        # Append top pick to Firebase
        if top3:
            top_pick = {
                "Restaurant": top3[0]["Restaurant"],
                "Rating": top3[0]["Rating"],
                "Address": top3[0]["Address"],
                "Google Maps Link": top3[0]["Google Maps Link"],
                "Food": food,
                "Location": location
            }
            append_history(top_pick)

# --------- DEEP LEARNING PAGE ---------
elif st.session_state.page == "Deep Learning":
    st.title("🤖 Deep Learning Explained")
    st.markdown("""
    This app uses **BERT sentiment analysis** to evaluate restaurant reviews and recommend the best places.

    ### Workflow:
    - Search nearby restaurants using **Foursquare API**.
    - Retrieve user reviews ("tips") and photos.
    - Analyze sentiment using **BERT model**.
    - Rank restaurants and show top recommendations.
    """)

# --------- HISTORY PAGE ---------
elif st.session_state.page == "History":
    st.title("📚 Recommendation History")
    history_data = read_history()
    if not history_data:
        st.info("No history yet. Make some recommendations!")
    else:
        df_hist = pd.DataFrame(history_data).drop(columns=['id','timestamp'], errors='ignore')
        if 'Google Maps Link' in df_hist.columns:
            df_hist['Map'] = df_hist['Google Maps Link'].apply(lambda x: f"[📍 View on Map]({x})")
        df_hist.index += 1
        st.dataframe(df_hist, use_container_width=True)

# --------- ABOUT PAGE ---------
elif st.session_state.page == "About":
    st.title("ℹ️ About This App")
    st.markdown("""
    **AI Restaurant Recommender** helps you find top restaurants using:

    - Foursquare API for places and reviews
    - BERT sentiment analysis
    - Firebase Firestore for history tracking
    - Google Maps links for navigation
    """)

# --------- FOOTER ---------
st.markdown('<div class="custom-footer">© 2025 AI Restaurant Recommender</div>', unsafe_allow_html=True)
