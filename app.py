import streamlit as st
import pandas as pd
import requests
import urllib.parse
from transformers import pipeline
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime
from streamlit_javascript import st_javascript

# Page config
st.set_page_config(page_title="🍽️ Restaurant Recommender", layout="wide")

# Background & overlay CSS
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
    top: 0; left: 0; right: 0; bottom: 0;
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

# Reverse geocode function using OpenStreetMap Nominatim API
def reverse_geocode(lat, lon):
    url = f"https://nominatim.openstreetmap.org/reverse?format=jsonv2&lat={lat}&lon={lon}"
    try:
        res = requests.get(url, headers={'User-Agent': 'streamlit-app'})
        if res.status_code == 200:
            data = res.json()
            address = data.get("address", {})
            city = address.get("city") or address.get("town") or address.get("village") or ""
            state = address.get("state") or ""
            country = address.get("country") or ""
            loc = f"{city or state}, {country}".strip(", ")
            return loc
        else:
            return ""
    except Exception as e:
        return ""

# Load sentiment analysis model (cached)
@st.cache_resource(show_spinner=False)
def get_classifier():
    return pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

# Initialize Firebase once
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
        docs = db.collection("recommendations").where("Restaurant", "==", data_dict.get("Restaurant")) \
                                              .where("Food", "==", food) \
                                              .where("Location", "==", location) \
                                              .stream()
        if len(list(docs)) > 0:
            return
        data_dict["timestamp"] = datetime.now()
        db.collection("recommendations").add(data_dict)
        st.success("New recommendation saved to history!")
    except Exception as e:
        st.error(f"Error saving to Firebase: {e}")

# Session state initialization
if "page" not in st.session_state:
    st.session_state.page = "Recommend"

# Sidebar navigation
with st.sidebar:
    st.markdown("## 🍽️ Menu")
    if st.button("Recommend"):
        st.session_state.page = "Recommend"
    if st.button("Deep Learning"):
        st.session_state.page = "Deep Learning"
    if st.button("History"):
        st.session_state.page = "History"
    if st.button("About"):
        st.session_state.page = "About"

# Main app pages
if st.session_state.page == "Recommend":
    st.title("🍽️ AI Restaurant Recommender")
    st.markdown("Find top-rated restaurants near you using **Foursquare** and **AI sentiment analysis** of real user reviews.")

    if "results" not in st.session_state:
        st.session_state.results = None
        st.session_state.df = None

    # Food input
    food = st.text_input("🍕 Food Type", placeholder="e.g., Sushi, Jollof, Pizza", key="food_input")

    # Location input + Detect Location button
    col1, col2 = st.columns([3,1])
    with col1:
        location = st.text_input("📍 Location", placeholder="e.g., Lagos, Nigeria", key="location_input")
    with col2:
        if st.button("📍 Detect My Location"):
            coords = st_javascript("navigator.geolocation.getCurrentPosition(pos => pos.coords.latitude + ',' + pos.coords.longitude)")
            if coords:
                lat_str, lon_str = coords.split(",")
                lat, lon = float(lat_str), float(lon_str)
                detected_loc = reverse_geocode(lat, lon)
                if detected_loc:
                    st.session_state.location_input = detected_loc
                    st.experimental_rerun()
                else:
                    st.warning("Could not determine location name from coordinates.")
            else:
                st.warning("Location access denied or unavailable.")

    api_key = st.secrets.get("FOURSQUARE_API_KEY", "")

    if st.button("🔍 Search"):
        food = st.session_state.get("food_input", "")
        location = st.session_state.get("location_input", "")

        if not food or not location:
            st.warning("⚠️ Please enter both a food type and location.")
        elif not api_key:
            st.error("❌ Foursquare API key is missing.")
        else:
            st.session_state.results = None
            st.session_state.df = None

            with st.spinner("Searching and analyzing reviews..."):
                headers = {"accept": "application/json", "Authorization": api_key}
                params = {"query": food, "near": location, "limit": 20}
                res = requests.get("https://api.foursquare.com/v3/places/search", headers=headers, params=params)
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

    if st.session_state.results:
        st.divider()
        st.subheader("📊 Restaurants Search Results and Ratings")
        st.dataframe(st.session_state.df, use_container_width=True)

        reviewed_restaurants = [r for r in st.session_state.results if r["Reviews"] > 0]
        top3 = sorted(reviewed_restaurants, key=lambda x: x["Rating"], reverse=True)[:3] if reviewed_restaurants else []
        
        st.divider()
        st.subheader("🏅 AI (Deep Learning) Top Picks")

        cols = st.columns(3)
        medals = ["🥇 1st", "🥈 2nd", "🥉 3rd"]
        colors = ["#FFD700", "#C0C0C0", "#CD7F32"]

        for i, (col, medal, color) in enumerate(zip(cols, medals, colors)):
            if i < len(top3):
                r = top3[i]
                with col:
                    st.markdown(f"""
                        <div style="background-color: {color}; border-radius: 15px; padding: 20px; text-align: center; box-shadow: 0 4px 8px rgba(0,0,0,0.2); color: black; font-weight: bold;">
                            <div style="font-size: 22px; margin-bottom: 10px;">{medal}</div>
                            <div style="font-size: 18px; margin-bottom: 8px;">{r['Restaurant']}</div>
                            <div style="font-size: 15px; margin-bottom: 8px;">{r['Address']}</div>
                            <div style="font-size: 16px;">{r['Stars']} ({r['Rating']})</div>
                            <div style="margin-top: 10px;">
                                <a href="{r['Google Maps Link']}" target="_blank" class="map-link">📍 locate restaurant</a>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

        st.divider()
        st.subheader("🖼️ Gallery Pick")

        restaurants_with_images = [r for r in st.session_state.results if r["Image"]]
        gallery_cols = st.columns(3)
        
        for idx, r in enumerate(sorted(restaurants_with_images, key=lambda x: x["Rating"] if x["Rating"] > 0 else 0, reverse=True)):
            with gallery_cols[idx % 3]:
                st.markdown(f"""
                    <div class="gallery-img-container">
                        <img src="{r['Image']}" class="gallery-img" />
                    </div>
                    <div class="gallery-caption">
                        <strong>{r['Restaurant']}</strong><br>
                        {'⭐ ' + str(r['Rating']) if r['Rating'] > 0 else 'No reviews'}<br>
                        <a href="{r['Google Maps Link']}" target="_blank" class="map-link">📍 View on Map</a>
                    </div>
                """, unsafe_allow_html=True)

        st.divider()
        if reviewed_restaurants:
            top = max(reviewed_restaurants, key=lambda x: x["Rating"])
            st.metric(label="🏆 Top Pick", value=top["Restaurant"], delta=f"{top['Rating']} ⭐")

            top_pick = {
                "Restaurant": top["Restaurant"],
                "Rating": top["Rating"],
                "Address": top["Address"],
                "Google Maps Link": top["Google Maps Link"],
                "Food": food,
                "Location": location
            }
            append_history(top_pick)
        else:
            st.warning("No restaurants with reviews found to select a top pick.")

        st.divider()
        st.subheader("📸 Restaurant Highlights")

        cols = st.columns(2)
        for idx, r in enumerate(sorted(st.session_state.results, key=lambda x: x["Rating"] if x["Rating"] > 0 else 0, reverse=True)):
            with cols[idx % 2]:
                st.markdown(f"### {r['Restaurant']}")
                st.markdown(f"**📍 Address:** {r['Address']}")
                st.markdown(f"[locate restaurant]({r['Google Maps Link']})", unsafe_allow_html=True)
                st.markdown(f"**⭐ Rating:** {r['Rating']} ({r['Reviews']} reviews)" if r['Reviews'] > 0 else "**⭐ Rating:** No reviews")
                if r["Image"]:
                    st.markdown(f"""
                        <div style="width: 100%; height: 220px; overflow: hidden; border-radius: 10px; margin-bottom: 10px;">
                            <img src="{r['Image']}" style="width: 100%; height: 100%; object-fit: cover;" />
                        </div>
                    """, unsafe_allow_html=True)
                st.markdown("💬 **Reviews:**")
                for tip in r["Tips"]:
                    st.markdown(f"• _{tip}_")
                st.markdown("---")

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

    Feel free to explore the Recommend tab and try it yourself!
    """)

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

elif st.session_state.page == "About":
    st.title("ℹ️ About This App")
    st.markdown("""
    **AI Restaurant Recommender** is a Streamlit web app designed to help you discover top restaurants based on your food cravings and location using:

    - [Foursquare API](https://developer.foursquare.com/) for places and user reviews.
    - State-of-the-art BERT-based sentiment analysis model from Hugging Face.
    - Firebase Firestore to save and track your recommendation history.
    - Google Maps integration for easy navigation to recommended restaurants.

    --- 
    _Powered by OpenAI and Streamlit._
    """)

st.markdown('<div class="custom-footer">© 2025 AI Restaurant Recommender</div>', unsafe_allow_html=True)
