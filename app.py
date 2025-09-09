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

# ------------------------ PAGE CONFIG ------------------------
st.set_page_config(page_title="🍽️ Restaurant Recommender", layout="wide")

# ------------------------ STYLES ------------------------
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

# ------------------------ SENTIMENT MODEL ------------------------
@st.cache_resource(show_spinner=False)
def get_classifier():
    return pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

classifier = get_classifier()

# ------------------------ FIREBASE INIT ------------------------
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

# ------------------------ FIREBASE HISTORY FUNCTIONS ------------------------
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

# ------------------------ SESSION STATE ------------------------
if "page" not in st.session_state:
    st.session_state.page = "Recommend"

# ------------------------ SIDEBAR ------------------------
with st.sidebar:
    st.markdown("## 🧭 Menu")
    if st.button("Recommend"): st.session_state.page = "Recommend"
    if st.button("Deep Learning"): st.session_state.page = "Deep Learning"
    if st.button("History"): st.session_state.page = "History"
    if st.button("About"): st.session_state.page = "About"

# ------------------------ API CALL CACHE ------------------------
@st.cache_data(ttl=3600)
def fetch_foursquare(url, headers, params=None):
    return requests.get(url, headers=headers, params=params).json()

# ------------------------ PAGES ------------------------
# -------- PAGE: Recommend --------
if st.session_state.page == "Recommend":
    st.title("🍽️ AI Restaurant Recommender")
    st.markdown("Find top-rated restaurants near you using **Foursquare** and **AI sentiment analysis** of real user reviews.")

    col1, _ = st.columns([1,1])
    with col1: food = st.text_input("🍕 Food Type", placeholder="e.g., Sushi, Jollof, Pizza")
    col1, _ = st.columns([1,1])
    with col1: location = st.text_input("📍 Location", placeholder="e.g., Lagos, Nigeria")

    SERVICE_KEY = st.secrets.get("FOURSQUARE_SERVICE_KEY", "")
    HEADERS = {
        "accept": "application/json",
        "Authorization": f"Bearer {SERVICE_KEY}",
        "X-Places-Api-Version": "2025-06-17"
    }

    if st.button("🔍 Search"):
        if not food or not location:
            st.warning("⚠️ Please enter both a food type and location.")
        elif not SERVICE_KEY:
            st.error("❌ Foursquare Service Key is missing.")
        else:
            with st.spinner("Searching and analyzing reviews..."):
                params = {"query": food, "near": location, "limit": 20}
                res = fetch_foursquare("https://places-api.foursquare.com/places/search", HEADERS, params)
                restaurants = res.get("results", [])

                if not restaurants:
                    st.error("❌ No restaurants found. Try different search terms.")
                else:
                    results = []
                    for r in restaurants:
                        fsq_id = r.get('fsq_id') or r.get('fsq_place_id')
                        name = r.get('name', 'Unknown')
                        address = r.get('location', {}).get('formatted_address', 'Unknown')
                        maps_link = f"https://www.google.com/maps/search/?api=1&query={urllib.parse.quote_plus(name + ', ' + address)}"

                        # Tips
                        tips_data = fetch_foursquare(f"https://places-api.foursquare.com/places/{fsq_id}/tips", HEADERS)
                        review_texts = [tip.get("text","") for tip in tips_data[:5]] if tips_data else []

                        # Sentiment
                        sentiments = []
                        for tip in review_texts:
                            res = classifier(tip[:512])[0]
                            stars = int(res["label"].split()[0])
                            sentiments.append(stars)

                        # Photos
                        photos_data = fetch_foursquare(f"https://places-api.foursquare.com/places/{fsq_id}/photos", HEADERS)
                        photo_url = f"{photos_data[0]['prefix']}original{photos_data[0]['suffix']}" if photos_data else ""

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

                    if results:
                        df = pd.DataFrame([{"Restaurant": r["Restaurant"], "Address": r["Address"], 
                                            "Average Rating": r["Rating"], "Stars": r["Stars"], 
                                            "Reviews": r["Reviews"]} for r in results])
                        df.index += 1
                        st.session_state.results = results
                        st.session_state.df = df

    # Display results
    if st.session_state.get("results"):
        st.divider()
        st.subheader("📊 Restaurants Search Results")
        st.dataframe(st.session_state.df, use_container_width=True)

        # ---------------- ANALYSIS ----------------
        st.divider()
        st.subheader("📈 Recommendation Analysis")
        analysis_df = pd.DataFrame(st.session_state.results)
        if analysis_df['Rating'].sum() > 0:
            tab1, tab2 = st.tabs(["Top Categories", "Review Insights"])
            with tab1:
                analysis_df['Category'] = analysis_df['Restaurant'].apply(lambda x: ' '.join([w for w in x.split() if w.istitle()][:2]))
                category_df = analysis_df.groupby('Category').agg({'Rating':'mean','Restaurant':'count'}).rename(columns={'Restaurant':'Count'}).sort_values('Rating',ascending=False)
                if not category_df.empty:
                    fig = px.bar(category_df.head(10), x=category_df.head(10).index, y='Rating', title='Top Restaurant Categories by Average Rating', color='Rating', color_continuous_scale='thermal')
                    st.plotly_chart(fig,use_container_width=True)
            with tab2:
                all_reviews = [review for sublist in analysis_df['Tips'] for review in sublist if review!="No reviews available"]
                if all_reviews:
                    text = ' '.join(all_reviews)
                    wordcloud = WordCloud(width=800, height=400, background_color='black').generate(text)
                    plt.figure(figsize=(10,5))
                    plt.imshow(wordcloud, interpolation='bilinear')
                    plt.axis("off")
                    st.pyplot(plt)
                    st.markdown("### 📝 Longest Reviews")
                    for i, review in enumerate(sorted(all_reviews,key=len,reverse=True)[:3],1):
                        st.markdown(f"{i}. {review[:300]}..." if len(review)>300 else f"{i}. {review}")
                else:
                    st.warning("No reviews available for analysis")
        else:
            st.info("No rating data available for analysis in current search results")

        # ---------------- TOP PICKS ----------------
        reviewed = [r for r in st.session_state.results if r["Reviews"]>0]
        top3 = sorted(reviewed, key=lambda x: x["Rating"], reverse=True)[:3] if reviewed else []

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
                        <div style="background-color:{color}; border-radius:15px; padding:20px; text-align:center; box-shadow:0 4px 8px rgba(0,0,0,0.2); color:black; font-weight:bold;">
                            <div style="font-size:22px; margin-bottom:10px;">{medal}</div>
                            <div style="font-size:18px; margin-bottom:8px;">{r['Restaurant']}</div>
                            <div style="font-size:15px; margin-bottom:8px;">{r['Address']}</div>
                            <div style="font-size:16px;">{r['Stars']} ({r['Rating']})</div>
                            <div style="margin-top:10px;">
                                <a href="{r['Google Maps Link']}" target="_blank" class="map-link">📍 locate restaurant</a>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

        # ---------------- GALLERY ----------------
        st.divider()
        st.subheader("🖼️ Gallery Pick")
        with_images = [r for r in st.session_state.results if r["Image"]]
        gallery_cols = st.columns(3)
        for idx, r in enumerate(sorted(with_images, key=lambda x:x["Rating"],reverse=True)):
            with gallery_cols[idx%3]:
                st.markdown(f"""
                    <div class="gallery-img-container"><img src="{r['Image']}" class="gallery-img"/></div>
                    <div class="gallery-caption"><strong>{r['Restaurant']}</strong><br>{'⭐ ' + str(r['Rating']) if r['Rating']>0 else 'No reviews'}<br>
                    <a href="{r['Google Maps Link']}" target="_blank" class="map-link">📍 View on Map</a></div>
                """, unsafe_allow_html=True)

        # ---------------- TOP PICK HISTORY ----------------
        st.divider()
        if reviewed:
            top = max(reviewed, key=lambda x:x["Rating"])
            st.metric(label="🏆 Top Pick", value=top["Restaurant"], delta=f"{top['Rating']} ⭐")
            append_history({"Restaurant":top["Restaurant"],"Rating":top["Rating"],"Address":top["Address"],"Google Maps Link":top["Google Maps Link"],"Food":food,"Location":location})
        else:
            st.warning("No restaurants with reviews found to select a top pick.")

# -------- PAGE: Deep Learning --------
elif st.session_state.page == "Deep Learning":
    st.title("🤖 Deep Learning Explained")
    st.markdown("""
    This app uses **BERT-based sentiment analysis** to evaluate restaurant reviews and provide AI-driven recommendations.

    - Fetches restaurants from **Foursquare API**.
    - Retrieves user reviews.
    - Uses BERT to predict sentiment.
    - Calculates average rating and ranks restaurants.
    """)

# -------- PAGE: History --------
elif st.session_state.page == "History":
    st.title("📚 Recommendation History")
    history_data = read_history()
    if not history_data:
        st.info("No history available yet.")
    else:
        df_hist = pd.DataFrame(history_data)
        df_hist = df_hist.drop(columns=['id','timestamp'],errors='ignore')
        if 'Google Maps Link' in df_hist.columns:
            df_hist['Map'] = df_hist['Google Maps Link'].apply(lambda x:f"[📍 View on Map]({x})")
        df_hist.index += 1
        st.dataframe(df_hist,use_container_width=True)

# -------- PAGE: About --------
elif st.session_state.page == "About":
    st.title("ℹ️ About This App")
    st.markdown("""
    **AI Restaurant Recommender** helps you discover top restaurants using:

    - [Foursquare API](https://developer.foursquare.com/)
    - BERT sentiment analysis
    - Firebase Firestore for history
    - Google Maps links
    """)
    
# ------------------------ FOOTER ------------------------
st.markdown('<div class="custom-footer">© 2025 AI Restaurant Recommender</div>', unsafe_allow_html=True)
