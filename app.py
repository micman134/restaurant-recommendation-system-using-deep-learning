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
import json

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
    
    /* Debug section styling */
    .debug-section {
        background-color: #2d2d2d;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border-left: 4px solid #ff6b6b;
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
    try:
        return pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
    except Exception as e:
        st.error(f"Failed to load sentiment analysis model: {e}")
        return None

# Initialize Firebase
try:
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
except Exception as e:
    st.error(f"Firebase initialization failed: {e}")
    db = None

def read_history():
    if not db:
        return []
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
    if not db:
        return
        
    food = data_dict.get("Food", "").strip()
    location = data_dict.get("Location", "").strip()

    if not food or not location:
        return

    try:
        # Check for duplicate entry
        docs = db.collection("recommendations") \
                 .where("Restaurant", "==", data_dict.get("Restaurant")) \
                 .where("Food", "==", food) \
                 .where("Location", "==", location) \
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
if "debug_mode" not in st.session_state:
    st.session_state.debug_mode = False
if "results" not in st.session_state:
    st.session_state.results = None
if "df" not in st.session_state:
    st.session_state.df = None

# Sidebar navigation
with st.sidebar:
    st.markdown("## 🍕 Menu")
    if st.button("Recommend"):
        st.session_state.page = "Recommend"
    if st.button("Deep Learning"):
        st.session_state.page = "Deep Learning"
    if st.button("History"):
        st.session_state.page = "History"
    if st.button("About"):
        st.session_state.page = "About"
    
    st.divider()
    st.session_state.debug_mode = st.checkbox("🔧 Debug Mode", value=False)

# -------- PAGE: Recommend --------
if st.session_state.page == "Recommend":
    st.title("🍽️ AI Restaurant Recommender")
    st.markdown("Find top-rated restaurants near you using **Foursquare** and **AI sentiment analysis** of real user reviews.")

    col1, col2 = st.columns([2, 1])
    with col1:
        food = st.text_input("🍕 Food Type", placeholder="e.g., Sushi, Jollof, Pizza")
    with col2:
        location = st.text_input("📍 Location", placeholder="e.g., Lagos, Nigeria")

    api_key = st.secrets.get("FOURSQUARE_API_KEY", "")

    # Test button for debugging
    if st.session_state.debug_mode:
        if st.button("🛠️ Test API with Sample Data"):
            test_food = "pizza"
            test_location = "New York"
            
            st.markdown("### 🧪 API Test Results")
            st.write(f"Testing with: Food='{test_food}', Location='{test_location}'")
            
            headers = {"accept": "application/json", "Authorization": api_key}
            params = {"query": test_food, "near": test_location, "limit": 5}
            
            try:
                res = requests.get("https://api.foursquare.com/v3/places/search", 
                                  headers=headers, params=params, timeout=10)
                
                st.write(f"**Status Code:** {res.status_code}")
                st.write(f"**Response Headers:** {dict(res.headers)}")
                
                if res.status_code == 200:
                    data = res.json()
                    restaurants = data.get("results", [])
                    st.write(f"**Found {len(restaurants)} restaurants**")
                    for i, r in enumerate(restaurants, 1):
                        st.write(f"{i}. {r.get('name', 'Unknown')} - {r.get('location', {}).get('formatted_address', 'No address')}")
                else:
                    st.write(f"**Error Response:** {res.text}")
                    
            except Exception as e:
                st.error(f"Test failed: {str(e)}")

    if st.button("🔍 Search Restaurants"):
        if not food or not location:
            st.warning("⚠️ Please enter both a food type and location.")
        elif not api_key:
            st.error("❌ Foursquare API key is missing. Please check your secrets configuration.")
        else:
            st.session_state.results = None
            st.session_state.df = None

            with st.spinner("Searching and analyzing reviews..."):
                if st.session_state.debug_mode:
                    st.markdown("### 🔧 Debug Information")
                    st.write(f"**Food:** '{food}'")
                    st.write(f"**Location:** '{location}'")
                    st.write(f"**API Key Present:** {bool(api_key)}")
                    st.write(f"**API Key Starts With:** {api_key[:10]}..." if api_key else "No API key")
                
                headers = {"accept": "application/json", "Authorization": api_key}
                params = {"query": food, "near": location, "limit": 20, "fields": "fsq_id,name,location"}
                
                try:
                    res = requests.get("https://api.foursquare.com/v3/places/search", 
                                      headers=headers, params=params, timeout=15)
                    
                    if st.session_state.debug_mode:
                        st.write(f"**Status Code:** {res.status_code}")
                        st.write(f"**Response Time:** {res.elapsed.total_seconds():.2f}s")
                    
                    if res.status_code != 200:
                        error_msg = f"❌ API Error: {res.status_code} - {res.reason}"
                        if res.status_code == 401:
                            error_msg += " (Invalid API Key)"
                        elif res.status_code == 403:
                            error_msg += " (Forbidden - check API permissions)"
                        st.error(error_msg)
                        
                        if st.session_state.debug_mode:
                            st.write(f"**Error Response:** {res.text[:200]}...")
                        restaurants = []
                    else:
                        data = res.json()
                        restaurants = data.get("results", [])
                        
                        if st.session_state.debug_mode:
                            st.write(f"**Raw API Response:**")
                            st.json(data)
                            st.write(f"**Found {len(restaurants)} restaurants**")
                        
                        if not restaurants:
                            st.warning("ℹ️ No restaurants found in the API response. Try a different location or cuisine.")
                            
                except requests.exceptions.Timeout:
                    st.error("⏰ Request timeout. The API is taking too long to respond.")
                    restaurants = []
                except requests.exceptions.ConnectionError:
                    st.error("🔌 Connection error. Please check your internet connection.")
                    restaurants = []
                except Exception as e:
                    st.error(f"❌ Unexpected error: {str(e)}")
                    restaurants = []

                # Process restaurants if found
                if restaurants:
                    classifier = get_classifier()
                    if classifier is None:
                        st.error("❌ Sentiment analysis model failed to load. Cannot analyze reviews.")
                        results = []
                    else:
                        results = []
                        progress_bar = st.progress(0)
                        
                        for i, r in enumerate(restaurants):
                            fsq_id = r['fsq_id']
                            name = r['name']
                            address = r['location'].get('formatted_address', 'Address not available')
                            
                            # Create Google Maps link
                            maps_query = urllib.parse.quote_plus(f"{name}, {address}")
                            maps_link = f"https://www.google.com/maps/search/?api=1&query={maps_query}"

                            # Get tips/reviews
                            tips = []
                            try:
                                tips_url = f"https://api.foursquare.com/v3/places/{fsq_id}/tips"
                                tips_res = requests.get(tips_url, headers=headers, timeout=10)
                                if tips_res.status_code == 200:
                                    tips = tips_res.json()
                            except:
                                tips = []

                            review_texts = [tip["text"] for tip in tips[:5]] if tips else []

                            # Sentiment analysis
                            sentiments = []
                            for tip in review_texts:
                                try:
                                    result = classifier(tip[:512])[0]
                                    stars = int(result["label"].split()[0])
                                    sentiments.append(stars)
                                except:
                                    continue

                            # Get photos
                            photo_url = ""
                            try:
                                photo_api = f"https://api.foursquare.com/v3/places/{fsq_id}/photos"
                                photo_res = requests.get(photo_api, headers=headers, timeout=10)
                                if photo_res.status_code == 200:
                                    photos = photo_res.json()
                                    if photos:
                                        photo = photos[0]
                                        photo_url = f"{photo['prefix']}original{photo['suffix']}"
                            except:
                                pass

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
                            
                            progress_bar.progress((i + 1) / len(restaurants))

                        progress_bar.empty()

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
                        st.warning("No restaurants could be processed successfully.")
                else:
                    st.error("❌ No restaurants found. Please try:")
                    st.write("- Different location (e.g., 'New York' instead of 'NYC')")
                    st.write("- Broader food category (e.g., 'Italian' instead of 'specific dish')")
                    st.write("- Check if the location exists on Foursquare")

    # Display results if available
    if st.session_state.results:
        st.divider()
        st.subheader("📊 Restaurants Search Results and Ratings")
        st.dataframe(st.session_state.df, use_container_width=True)

        # Analysis Section
        st.divider()
        st.subheader("📈 Recommendation Analysis")
        
        analysis_df = pd.DataFrame(st.session_state.results)
        
        if analysis_df['Rating'].sum() > 0:
            tab1, tab2 = st.tabs(["Top Categories", "Review Insights"])
            
            with tab1:
                analysis_df['Category'] = analysis_df['Restaurant'].apply(lambda x: ' '.join([w for w in x.split() if w.isupper() or w.istitle()][:2]))
                category_df = analysis_df.groupby('Category').agg({
                    'Rating': 'mean',
                    'Restaurant': 'count'
                }).rename(columns={'Restaurant': 'Count'}).sort_values('Rating', ascending=False)
                
                if not category_df.empty:
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
                st.markdown("### 💬 Review Sentiment Highlights")
                all_reviews = [review for sublist in analysis_df['Tips'] for review in sublist if review != "No reviews available"]
                
                if all_reviews:
                    text = ' '.join(all_reviews)
                    wordcloud = WordCloud(width=800, height=400, background_color='black').generate(text)
                    
                    plt.figure(figsize=(10, 5))
                    plt.imshow(wordcloud, interpolation='bilinear')
                    plt.axis("off")
                    st.pyplot(plt)
                    
                    st.markdown("### 📝 Longest Reviews")
                    longest_reviews = sorted(all_reviews, key=len, reverse=True)[:3]
                    for i, review in enumerate(longest_reviews, 1):
                        st.markdown(f"{i}. {review[:300]}..." if len(review) > 300 else f"{i}. {review}")
                else:
                    st.warning("No reviews available for analysis")

        # Top Picks Section
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

        # Gallery Section
        st.divider()
        st.subheader("🖼️ Gallery Pick")
        restaurants_with_images = [r for r in st.session_state.results if r["Image"]]
        
        if restaurants_with_images:
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
        else:
            st.info("No restaurant images available for gallery display.")

        # Top Pick Metric
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

        # Restaurant Highlights
        st.divider()
        st.subheader("📸 Restaurant Highlights")

        cols = st.columns(2)
        for idx, r in enumerate(sorted(st.session_state.results, key=lambda x: x["Rating"] if x["Rating"] > 0 else 0, reverse=True)):
            with cols[idx % 2]:
                st.markdown(f"### {r['Restaurant']}")
                st.markdown(f"**📍 Address:** {r['Address']}")
                st.markdown(f"[📍 locate restaurant]({r['Google Maps Link']})", unsafe_allow_html=True)
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

    --- 
    _Powered by OpenAI and Streamlit._
    """)

# Footer
st.markdown('<div class="custom-footer">© 2025 AI Restaurant Recommender</div>', unsafe_allow_html=True)
