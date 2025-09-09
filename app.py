import streamlit as st
import requests
import pandas as pd
from transformers import pipeline
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime
import urllib.parse
import matplotlib.pyplot as plt

# --------- PAGE CONFIGURATION ---------
st.set_page_config(page_title="🍽️ Restaurant Recommender", layout="wide")

# --------- CSS STYLING ---------
st.markdown("""
<style>
.stApp { background-image: url("https://images.unsplash.com/photo-1517248135467-4c7edcad34c4"); background-size: cover; background-position: center; background-attachment: fixed;}
.stApp:before {content: ""; position: absolute; top: 0; left: 0; right: 0; bottom: 0; background: rgba(0,0,0,0.85); z-index: 0;}
#MainMenu, footer, header {visibility: hidden;}
.custom-footer {text-align: center; font-size: 14px; margin-top: 50px; padding: 20px; color: #aaa;}
.gallery-img-container { width: 100%; height: 250px; overflow: hidden; border-radius: 10px; margin-bottom: 10px;}
.gallery-img { width: 100%; height: 100%; object-fit: cover; }
.gallery-caption { text-align: center; margin-top: 5px; }
.map-link { color: #4CAF50 !important; text-decoration: none; font-weight: bold; }
.map-link:hover { text-decoration: underline; }
</style>
""", unsafe_allow_html=True)

# --------- SENTIMENT / TEXT GENERATION MODELS ---------
@st.cache_resource(show_spinner=False)
def get_sentiment_model():
    return pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

@st.cache_resource(show_spinner=False)
def get_review_generator():
    return pipeline("text-generation", model="gpt2")

sentiment_model = get_sentiment_model()
review_gen = get_review_generator()

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
        return [dict(doc.to_dict(), id=doc.id) for doc in docs]
    except Exception as e:
        st.error(f"Error reading from Firebase: {e}")
        return []

def append_history(data_dict):
    if not data_dict.get("Food") or not data_dict.get("Location"):
        return
    try:
        # check duplicates
        docs = db.collection("recommendations")\
                 .where("Restaurant","==",data_dict.get("Restaurant"))\
                 .where("Food","==",data_dict.get("Food"))\
                 .where("Location","==",data_dict.get("Location")).stream()
        if len(list(docs))>0: return
        data_dict["timestamp"] = datetime.now()
        db.collection("recommendations").add(data_dict)
    except Exception as e:
        st.error(f"Error saving to Firebase: {e}")

# --------- SESSION STATE ---------
if "page" not in st.session_state: st.session_state.page = "Recommend"

# --------- SIDEBAR ---------
with st.sidebar:
    st.markdown("## 🍽️ Menu")
    if st.button("Recommend"): st.session_state.page="Recommend"
    if st.button("Deep Learning"): st.session_state.page="Deep Learning"
    if st.button("History"): st.session_state.page="History"
    if st.button("About"): st.session_state.page="About"

# --------- GOOGLE API CONFIG ---------
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
CX = st.secrets["CX"]

# --------- RECOMMEND PAGE ---------
if st.session_state.page=="Recommend":
    st.title("🍽️ AI Restaurant Recommender")
    st.markdown("Find top restaurants using **Google Images** + AI-generated reviews.")

    food = st.text_input("🍕 Food Type", placeholder="e.g., Sushi, Jollof, Pizza")
    location = st.text_input("📍 Location", placeholder="e.g., Lagos, Nigeria")

    if st.button("🔍 Search"):
        if not food or not location:
            st.warning("Enter both food type and location")
        else:
            st.session_state.results = []
            query = f"{food} restaurants in {location}"

            # Search Google Images for restaurant pics
            img_url_list = []
            try:
                search_res = requests.get(
                    "https://www.googleapis.com/customsearch/v1",
                    params={
                        "key": GOOGLE_API_KEY,
                        "cx": CX,
                        "q": query,
                        "searchType": "image",
                        "num": 10
                    }
                ).json()
                img_url_list = [item["link"] for item in search_res.get("items",[])]
            except Exception as e:
                st.error(f"Error fetching images: {e}")

            results = []
            for i, img_url in enumerate(img_url_list):
                restaurant_name = f"{food} Restaurant {i+1}"
                # Generate review
                review = review_gen(f"Write a positive review for {restaurant_name} in {location}", max_length=80)[0]["generated_text"]
                # Sentiment
                rating = int(sentiment_model(review[:512])[0]["label"].split()[0])
                results.append({
                    "Restaurant": restaurant_name,
                    "Address": location,
                    "Image": img_url,
                    "Review": review,
                    "Rating": rating,
                    "Stars": "⭐"*rating if rating>0 else "No reviews",
                    "Google Maps Link": f"https://www.google.com/maps/search/?api=1&query={urllib.parse.quote_plus(restaurant_name+' '+location)}"
                })

            st.session_state.results = results

    # Display
    if st.session_state.results:
        for r in st.session_state.results:
            st.markdown(f"### {r['Restaurant']} ({r['Stars']})")
            st.markdown(f"[📍 View on Map]({r['Google Maps Link']})")
            st.image(r["Image"], width=300)
            st.markdown(f"**Review:** {r['Review']}")
            # save top pick to Firebase (optional)
            if r==st.session_state.results[0]:
                append_history({
                    "Restaurant": r["Restaurant"],
                    "Rating": r["Rating"],
                    "Address": r["Address"],
                    "Google Maps Link": r["Google Maps Link"],
                    "Food": food,
                    "Location": location
                })

# --------- HISTORY PAGE ---------
elif st.session_state.page=="History":
    st.title("📚 Recommendation History")
    history_data = read_history()
    if history_data:
        df_hist = pd.DataFrame(history_data).drop(columns=['id','timestamp'], errors='ignore')
        if 'Google Maps Link' in df_hist.columns:
            df_hist['Map'] = df_hist['Google Maps Link'].apply(lambda x: f"[📍 View on Map]({x})")
        df_hist.index += 1
        st.dataframe(df_hist, use_container_width=True)
    else:
        st.info("No history yet.")

# --------- ABOUT PAGE ---------
elif st.session_state.page=="About":
    st.title("ℹ️ About")
    st.markdown("""
    **AI Restaurant Recommender** now uses:
    - Google Images for restaurant pictures
    - GPT2 for AI-generated reviews
    - BERT sentiment analysis for ratings
    - Firebase Firestore for history
    """)

# --------- FOOTER ---------
st.markdown('<div class="custom-footer">© 2025 AI Restaurant Recommender</div>', unsafe_allow_html=True)
