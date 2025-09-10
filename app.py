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

# ----------------- CONFIGURATION -----------------
st.set_page_config(page_title="🍽️ Restaurant Recommender", layout="wide")

SERVICE_KEY = st.secrets.get("FOURSQUARE_SERVICE_KEY", "")

HEADERS = {
    "Authorization": f"Bearer {SERVICE_KEY}",
    "X-Places-Api-Version": "2025-06-17",
    "Accept": "application/json"
}

# ----------------- STYLES -----------------
st.markdown("""
<style>
.stApp { background-image: url("https://images.unsplash.com/photo-1517248135467-4c7edcad34c4"); background-size: cover; background-position: center; background-repeat: no-repeat; background-attachment: fixed; }
.stApp:before { content: ""; position: absolute; top:0; left:0; right:0; bottom:0; background: rgba(0,0,0,0.85); z-index:0; }
#MainMenu, footer, header {visibility: hidden;}
.custom-footer {text-align:center;font-size:14px;margin-top:50px;padding:20px;color:#aaa;}
.gallery-img-container {width:100%;height:250px;overflow:hidden;border-radius:10px;margin-bottom:10px;}
.gallery-img {width:100%;height:100%;object-fit:cover;}
.gallery-caption {text-align:center;margin-top:5px;}
.map-link {color:#4CAF50 !important;text-decoration:none;font-weight:bold;}
.map-link:hover {text-decoration:underline;}
.stTabs [data-baseweb="tab-list"] {gap:10px;}
.stTabs [data-baseweb="tab"] {padding:8px 16px;border-radius:8px 8px 0 0;}
.stTabs [aria-selected="true"] {background-color:#4CAF50;color:white;}
</style>
""", unsafe_allow_html=True)

# ----------------- SENTIMENT ANALYSIS -----------------
@st.cache_resource(show_spinner=False)
def get_classifier():
    return pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

# ----------------- FIREBASE -----------------
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
        return [dict(doc.to_dict(), id=doc.id) for doc in docs]
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
        if len(list(docs)) > 0: return
        data_dict["timestamp"] = datetime.now()
        db.collection("recommendations").add(data_dict)
    except Exception as e:
        st.error(f"Error saving to Firebase: {e}")

# ----------------- SESSION -----------------
if "page" not in st.session_state:
    st.session_state.page = "Recommend"

with st.sidebar:
    st.markdown("## 🍽️ Menu")
    if st.button("Recommend"): st.session_state.page = "Recommend"
    if st.button("Deep Learning"): st.session_state.page = "Deep Learning"
    if st.button("History"): st.session_state.page = "History"
    if st.button("About"): st.session_state.page = "About"

# ----------------- HELPER FUNCTIONS -----------------
def search_places(query, location, limit=20):
    url = "https://places-api.foursquare.com/places/search"
    params = {"query": query, "near": location, "limit": limit}
    res = requests.get(url, headers=HEADERS, params=params)
    return res.json().get("results", [])

def get_tips(fsq_place_id, limit=5):
    url = f"https://places-api.foursquare.com/places/{fsq_place_id}/tips"
    try:
        res = requests.get(url, headers=HEADERS)
        return [tip["text"] for tip in res.json()[:limit]]
    except: return []

def get_photo(fsq_place_id):
    url = f"https://places-api.foursquare.com/places/{fsq_place_id}/photos"
    try:
        res = requests.get(url, headers=HEADERS)
        photos = res.json()
        if photos:
            photo = photos[0]
            return f"{photo['prefix']}original{photo['suffix']}"
    except: return None

# ----------------- PAGES -----------------
if st.session_state.page == "Recommend":
    st.title("🍽️ AI Restaurant Recommender")
    st.markdown("Find top-rated restaurants near you using **Foursquare** and **AI sentiment analysis**.")

    col1, _ = st.columns([1, 1])
    with col1:
        food = st.text_input("🍕 Food Type", placeholder="e.g., Sushi, Jollof, Pizza")
    col1, _ = st.columns([1, 1])
    with col1:
        location = st.text_input("📍 Location", placeholder="e.g., Lagos, Nigeria")

    if st.button("🔍 Search"):
        if not food or not location:
            st.warning("⚠️ Please enter both a food type and location.")
        elif not SERVICE_KEY:
            st.error("❌ Foursquare Service Key is missing.")
        else:
            st.session_state.results = []
            with st.spinner("Searching and analyzing reviews..."):
                restaurants = search_places(food, location)
                if not restaurants:
                    st.error("❌ No restaurants found. Try different search terms.")
                else:
                    classifier = get_classifier()
                    results = []
                    for r in restaurants:
                        fsq_id = r['fsq_place_id']
                        name = r['name']
                        address = r['location'].get('formatted_address', 'Unknown')
                        maps_link = f"https://www.google.com/maps/search/?api=1&query={urllib.parse.quote_plus(name + ', ' + address)}"
                        review_texts = get_tips(fsq_id)
                        sentiments = [int(classifier(t[:512])[0]["label"].split()[0]) for t in review_texts] if review_texts else []
                        photo_url = get_photo(fsq_id)
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
                    df = pd.DataFrame([{
                        "Restaurant": r["Restaurant"],
                        "Address": r["Address"],
                        "Average Rating": r["Rating"],
                        "Stars": r["Stars"],
                        "Reviews": r["Reviews"]
                    } for r in results])
                    df.index +=1
                    st.session_state.df = df

    if st.session_state.results:
        st.divider()
        st.subheader("📊 Restaurants Search Results and Ratings")
        st.dataframe(st.session_state.df, use_container_width=True)

        # ----- ANALYSIS -----
        st.divider()
        st.subheader("📈 Recommendation Analysis")
        analysis_df = pd.DataFrame(st.session_state.results)
        if analysis_df['Rating'].sum()>0:
            tab1, tab2 = st.tabs(["Top Categories", "Review Insights"])
            with tab1:
                analysis_df['Category'] = analysis_df['Restaurant'].apply(lambda x: ' '.join([w for w in x.split() if w.istitle()][:2]))
                category_df = analysis_df.groupby('Category').agg({'Rating':'mean','Restaurant':'count'}).rename(columns={'Restaurant':'Count'}).sort_values('Rating', ascending=False)
                if not category_df.empty:
                    fig = px.bar(category_df.head(10), x=category_df.head(10).index, y='Rating', color='Rating', color_continuous_scale='thermal', title='Top Restaurant Categories')
                    st.plotly_chart(fig, use_container_width=True)
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
                    longest_reviews = sorted(all_reviews, key=len, reverse=True)[:3]
                    for i, review in enumerate(longest_reviews,1):
                        st.markdown(f"{i}. {review[:300]}..." if len(review)>300 else f"{i}. {review}")
                else:
                    st.warning("No reviews available for analysis")
        else:
            st.info("No rating data available for analysis")

        # ----- TOP PICKS & GALLERY -----
        reviewed_restaurants = [r for r in st.session_state.results if r["Reviews"]>0]
        top3 = sorted(reviewed_restaurants, key=lambda x: x["Rating"], reverse=True)[:3] if reviewed_restaurants else []

        st.divider()
        st.subheader("🏅 AI (Deep Learning) Top Picks")
        cols = st.columns(3)
        medals = ["🥇 1st", "🥈 2nd", "🥉 3rd"]
        colors = ["#FFD700","#C0C0C0","#CD7F32"]
        for i, (col, medal, color) in enumerate(zip(cols,medals,colors)):
            if i<len(top3):
                r=top3[i]
                with col:
                    st.markdown(f"""
                    <div style="background-color:{color}; border-radius:15px; padding:20px; text-align:center; color:black; font-weight:bold;">
                    <div style="font-size:22px; margin-bottom:10px;">{medal}</div>
                    <div style="font-size:18px; margin-bottom:8px;">{r['Restaurant']}</div>
                    <div style="font-size:15px; margin-bottom:8px;">{r['Address']}</div>
                    <div style="font-size:16px;">{r['Stars']} ({r['Rating']})</div>
                    <div style="margin-top:10px;"><a href="{r['Google Maps Link']}" target="_blank" class="map-link">📍 locate restaurant</a></div>
                    </div>""", unsafe_allow_html=True)

        st.divider()
        st.subheader("🖼️ Gallery Pick")
        restaurants_with_images = [r for r in st.session_state.results if r["Image"]]
        gallery_cols = st.columns(3)
        for idx, r in enumerate(sorted(restaurants_with_images, key=lambda x: x["Rating"], reverse=True)):
            with gallery_cols[idx%3]:
                st.markdown(f"""
                <div class="gallery-img-container"><img src="{r['Image']}" class="gallery-img"/></div>
                <div class="gallery-caption"><strong>{r['Restaurant']}</strong><br>{'⭐ '+str(r['Rating']) if r['Rating']>0 else 'No reviews'}<br>
                <a href="{r['Google Maps Link']}" target="_blank" class="map-link">📍 View on Map</a></div>
                """, unsafe_allow_html=True)

        if reviewed_restaurants:
            top = max(reviewed_restaurants, key=lambda x: x["Rating"])
            st.metric(label="🏆 Top Pick", value=top["Restaurant"], delta=f"{top['Rating']} ⭐")
            append_history({"Restaurant": top["Restaurant"], "Rating": top["Rating"], "Address": top["Address"], "Google Maps Link": top["Google Maps Link"], "Food": food, "Location": location})

elif st.session_state.page == "Deep Learning":
    st.title("🤖 Deep Learning Explained")
    st.markdown("""...""")  # Keep your existing Deep Learning content

elif st.session_state.page == "History":
    st.title("📚 Recommendation History")
    history_data = read_history()
    if not history_data:
        st.info("No history available yet. Try making some recommendations first!")
    else:
        df_hist = pd.DataFrame(history_data).drop(columns=['id','timestamp'], errors='ignore')
        if 'Google Maps Link' in df_hist.columns:
            df_hist['Map'] = df_hist['Google Maps Link'].apply(lambda x: f"[📍 View on Map]({x})")
        df_hist.index += 1
        st.dataframe(df_hist, use_container_width=True)

elif st.session_state.page == "About":
    st.title("ℹ️ About This App")
    st.markdown("""...""")  # Keep your existing About content

st.markdown('<div class="custom-footer">© 2025 AI Restaurant Recommender</div>', unsafe_allow_html=True)
