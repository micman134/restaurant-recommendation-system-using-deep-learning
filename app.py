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
    
    /* Dark overlay */
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
    
    /* Keep all your existing styles below */
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
    
    /* Gallery image styling */
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
    
    /* Map link styling */
    .map-link {
        color: #4CAF50 !important;
        text-decoration: none;
        font-weight: bold;
    }
    .map-link:hover {
        text-decoration: underline;
    }
    
    /* Analysis tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
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
        # Check for duplicate entry using keyword arguments
        docs = db.collection("recommendations") \
                 .where(field_path="Restaurant", op_string="==", value=data_dict.get("Restaurant")) \
                 .where(field_path="Food", op_string="==", value=food) \
                 .where(field_path="Location", op_string="==", value=location) \
                 .stream()
        
        if len(list(docs)) > 0:
            return

        # Add timestamp
        data_dict["timestamp"] = datetime.now()
        
        # Add to Firestore
        db.collection("recommendations").add(data_dict)
    except Exception as e:
        st.error(f"Error saving to Firebase: {e}")

# Session state initialization
if "page" not in st.session_state:
    st.session_state.page = "Recommend"

# Sidebar navigation
with st.sidebar:
    st.markdown("## �️ Menu")
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
                        
                        # Create Google Maps link
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

        # ======== ANALYSIS SECTION ========
        st.divider()
        st.subheader("📈 Recommendation Analysis")
        
        # Create DataFrame from results
        analysis_df = pd.DataFrame(st.session_state.results)
        
        # Only show analysis if we have ratings
        if analysis_df['Rating'].sum() > 0:
            # Create tabs for different analysis views
            tab1, tab2= st.tabs(["Top Categories", "Review Insights"])
            
            
            with tab1:
                # Extract categories from food types
                analysis_df['Category'] = analysis_df['Restaurant'].apply(lambda x: ' '.join([w for w in x.split() if w.isupper() or w.istitle()][:2]))
                
                # Group by category
                category_df = analysis_df.groupby('Category').agg({
                    'Rating': 'mean',
                    'Restaurant': 'count'
                }).rename(columns={'Restaurant': 'Count'}).sort_values('Rating', ascending=False)
                
                if not category_df.empty:
                    # Category bar chart
                    fig3 = px.bar(category_df.head(10), 
                                 x=category_df.head(10).index,
                                 y='Rating',
                                 title='Top Restaurant Categories by Average Rating',
                                 color='Rating',
                                 color_continuous_scale='thermal')
                    st.plotly_chart(fig3, use_container_width=True)
                else:
                    st.warning("No category data available for visualization")
            
            with tab2:
                # Sentiment analysis of reviews
                st.markdown("### 💬 Review Sentiment Highlights")
                
                # Get all review texts
                all_reviews = [review for sublist in analysis_df['Tips'] for review in sublist if review != "No reviews available"]
                
                if all_reviews:
                    # Show word cloud of common terms
                    text = ' '.join(all_reviews)
                    wordcloud = WordCloud(width=800, height=400, background_color='black').generate(text)
                    
                    plt.figure(figsize=(10, 5))
                    plt.imshow(wordcloud, interpolation='bilinear')
                    plt.axis("off")
                    st.pyplot(plt)
                    
                    # Show longest reviews
                    st.markdown("### 📝 Longest Reviews")
                    longest_reviews = sorted(all_reviews, key=len, reverse=True)[:3]
                    for i, review in enumerate(longest_reviews, 1):
                        st.markdown(f"{i}. {review[:300]}..." if len(review) > 300 else f"{i}. {review}")
                else:
                    st.warning("No reviews available for analysis")
        else:
            st.info("No rating data available for analysis in current search results")

        # ======== CONTINUE WITH EXISTING CODE ========
        # Filter out restaurants with zero reviews for top picks
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

        # Gallery Pick Section
        st.divider()
        st.subheader("🖼️ Gallery Pick")

        # Filter out restaurants without images
        restaurants_with_images = [r for r in st.session_state.results if r["Image"]]
        
        # Create columns for the gallery
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

    Feel free to explore the Recommend tab and try it yourself!
    """)

# -------- PAGE: History --------
elif st.session_state.page == "History":
    st.title("📚 Recommendation History")

    history_data = read_history()
    if not history_data:
        st.info("No history available yet. Try making some recommendations first!")
    else:
        # Convert to DataFrame for nice display
        df_hist = pd.DataFrame(history_data)
        # Remove internal fields
        df_hist = df_hist.drop(columns=['id', 'timestamp'], errors='ignore')
        
        # Add map links if they exist in the data
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

    --- 
    _Powered by OpenAI and Streamlit._
    """)

# Footer
st.markdown('<div class="custom-footer">© 2025 AI Restaurant Recommender</div>', unsafe_allow_html=True)
