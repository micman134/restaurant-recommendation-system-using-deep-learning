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
import random
import time

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
    
    /* Image styling fixes */
    img {
        border-radius: 10px;
        margin-bottom: 10px;
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

# Load text generation model for reviews
@st.cache_resource(show_spinner=False)
def get_text_generator():
    try:
        # Using a smaller, faster model for text generation
        return pipeline("text-generation", model="distilgpt2", truncation=True)
    except:
        return None

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

# --------- IMAGE SERVICE FUNCTIONS ---------
def get_restaurant_image(restaurant_name, location, food_type):
    """Get restaurant image from multiple fallback services (no Google API)"""
    
    # List of reliable image services that don't require API keys
    image_services = [
        _get_unsplash_image,
        _get_picsum_image,
        _get_foodish_image,
        _get_placeholder_image
    ]
    
    # Try each service until we get a valid image
    for service in image_services:
        try:
            image_url = service(restaurant_name, location, food_type)
            if image_url and _validate_image_url(image_url):
                return image_url
        except:
            continue
    
    # Final fallback
    return "https://images.unsplash.com/photo-1517248135467-4c7edcad34c4?w=400&h=300&fit=crop"

def _get_unsplash_image(restaurant_name, location, food_type):
    """Get image from Unsplash (no API key needed)"""
    search_terms = [restaurant_name, food_type, "restaurant", "food", "dining", "cuisine"]
    random.shuffle(search_terms)
    query = ",".join(search_terms[:2])
    
    # Use different Unsplash endpoints for variety
    unsplash_options = [
        f"https://source.unsplash.com/400x300/?{query}",
        f"https://source.unsplash.com/featured/400x300/?{query}",
        f"https://source.unsplash.com/random/400x300/?{query}",
        f"https://images.unsplash.com/photo-{random.randint(1500000000000, 1600000000000)}?w=400&h=300&fit=crop"
    ]
    
    return random.choice(unsplash_options)

def _get_picsum_image(restaurant_name, location, food_type):
    """Get random food image from Picsum"""
    image_id = random.randint(1, 1000)
    return f"https://picsum.photos/400/300?random={image_id}"

def _get_foodish_image(restaurant_name, location, food_type):
    """Get food image from Foodish API"""
    try:
        food_types = ['pizza', 'burger', 'sushi', 'pasta', 'steak', 'chicken', 'salad', 'sandwich']
        food_choice = random.choice(food_types)
        return f"https://foodish-api.com/images/{food_choice}/{food_choice}{random.randint(1, 30)}.jpg"
    except:
        return ""

def _get_placeholder_image(restaurant_name, location, food_type):
    """Get placeholder image with restaurant name"""
    restaurant_initials = ''.join([word[0].upper() for word in restaurant_name.split()[:2]])
    color = random.choice(['4CAF50', '2196F3', 'FF5722', '9C27B0', '607D8B'])
    return f"https://via.placeholder.com/400x300/{color}/white?text={restaurant_initials}+{food_type}"

def _validate_image_url(url):
    """Validate that the image URL is accessible"""
    try:
        response = requests.head(url, timeout=3)
        return response.status_code == 200
    except:
        return False

# --------- REVIEW GENERATION WITH LOCAL AI MODEL ---------
def generate_ai_reviews(restaurant_name, food_type, rating, num_reviews):
    """Generate realistic reviews using local AI model"""
    reviews = []
    text_generator = get_text_generator()
    
    if text_generator is None:
        # Fallback to template-based reviews if AI model fails
        return generate_template_reviews(restaurant_name, food_type, rating, num_reviews)
    
    try:
        for i in range(num_reviews):
            # Create different prompts based on rating
            if rating >= 4.0:
                prompt = f"Write a positive restaurant review for {restaurant_name} that serves {food_type}:"
            elif rating >= 3.0:
                prompt = f"Write a mixed review for {restaurant_name} that serves {food_type}:"
            else:
                prompt = f"Write a negative review for {restaurant_name} that serves {food_type}:"
            
            # Generate review with AI
            generated = text_generator(
                prompt,
                max_length=80,
                num_return_sequences=1,
                temperature=0.9,
                do_sample=True,
                pad_token_id=50256
            )
            
            review_text = generated[0]['generated_text'].replace(prompt, '').strip()
            if review_text and len(review_text) > 10:
                reviews.append(review_text)
            else:
                # Fallback if AI generation fails
                reviews.append(f"Great {food_type.lower()} and excellent service at {restaurant_name}.")
                
    except Exception as e:
        return generate_template_reviews(restaurant_name, food_type, rating, num_reviews)
    
    return reviews

def generate_template_reviews(restaurant_name, food_type, rating, num_reviews):
    """Fallback template-based review generation"""
    reviews = []
    
    # Review templates based on rating
    if rating >= 4.0:
        templates = [
            f"Absolutely amazing {food_type.lower()}! {restaurant_name} has the best in town.",
            f"Five stars for {restaurant_name}! The {food_type.lower()} was perfection.",
            f"Exceptional dining experience. The {food_type.lower()} melted in my mouth.",
            f"{restaurant_name} never disappoints. Their {food_type.lower()} is always fresh.",
            f"Top-notch service and incredible {food_type.lower()}. Will definitely be back!",
            f"The ambiance at {restaurant_name} is perfect for enjoying {food_type.lower()}.",
            f"Highly recommend {restaurant_name} for their excellent {food_type.lower()}.",
            f"Best {food_type.lower()} I've had in a long time at {restaurant_name}."
        ]
    elif rating >= 3.0:
        templates = [
            f"Good {food_type.lower()} at {restaurant_name}, but room for improvement.",
            f"Solid choice for {food_type.lower()}. Service was friendly but a bit slow.",
            f"Decent experience. The {food_type.lower()} was good but not exceptional.",
            f"{restaurant_name} has potential. The {food_type.lower()} was tasty.",
            f"Average visit. The {food_type.lower()} was okay, nothing special.",
            f"Reasonable prices for {food_type.lower()} at {restaurant_name}.",
            f"Would visit again for the {food_type.lower()} but service could be better."
        ]
    else:
        templates = [
            f"Disappointing {food_type.lower()} at {restaurant_name}. Expected better.",
            f"Not impressed. The {food_type.lower()} was bland and overpriced.",
            f"{restaurant_name} needs improvement. The {food_type.lower()} was not fresh.",
            f"Poor experience. Service was slow and {food_type.lower()} was cold.",
            f"Would not recommend. The {food_type.lower()} was below average.",
            f"Very disappointed with the {food_type.lower()} quality at {restaurant_name}."
        ]
    
    # Select random templates
    selected_templates = random.sample(templates, min(num_reviews, len(templates)))
    
    # Add some variety to the reviews
    modifiers = [
        " The atmosphere was lovely.",
        " Great presentation!",
        " Very reasonable prices.",
        " Perfect for a casual dinner.",
        " Highly recommended!",
        " Could use more seasoning.",
        " Portions were generous.",
        " Will visit again soon.",
        " Nice ambiance overall.",
        " Good value for money.",
        " The staff was very attentive.",
        " Would definitely come back."
    ]
    
    for template in selected_templates:
        review = template + random.choice(modifiers)
        reviews.append(review)
    
    return reviews

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

    # Use SERVICE_KEY instead of API_KEY for new Foursquare API
    SERVICE_KEY = st.secrets.get("FOURSQUARE_SERVICE_KEY", "")

    if st.button("🔍 Search"):
        if not food or not location:
            st.warning("⚠️ Please enter both a food type and location.")
        elif not SERVICE_KEY:
            st.error("❌ Foursquare Service Key is missing.")
        else:
            st.session_state.results = None
            st.session_state.df = None

            with st.spinner("Searching restaurants, analyzing reviews, and generating AI content..."):
                # Updated headers for new Foursquare API
                headers = {
                    "Accept": "application/json",
                    "Authorization": f"Bearer {SERVICE_KEY}",
                    "X-Places-Api-Version": "2025-06-17"
                }
                
                # Updated endpoint and parameters
                params = {"query": food, "near": location, "limit": 12}
                try:
                    res = requests.get("https://places-api.foursquare.com/places/search", headers=headers, params=params, timeout=10)
                    
                    if res.status_code != 200:
                        st.error(f"❌ Foursquare API error: {res.status_code}")
                        restaurants = []
                    else:
                        data = res.json()
                        restaurants = data.get("results", [])
                except Exception as e:
                    st.error(f"❌ Foursquare API connection error: {e}")
                    restaurants = []

                if not restaurants:
                    st.error("❌ No restaurants found. Try different search terms.")
                else:
                    classifier = get_classifier()
                    results = []

                    for r in restaurants:
                        # Updated field name from fsq_id to fsq_place_id
                        fsq_place_id = r.get('fsq_place_id', '')
                        name = r.get('name', 'Unknown')
                        address = r.get('location', {}).get('formatted_address', 'Unknown')
                        
                        # Create Google Maps link
                        maps_query = urllib.parse.quote_plus(f"{name}, {address}")
                        maps_link = f"https://www.google.com/maps/search/?api=1&query={maps_query}"

                        # Get tips/reviews - updated endpoint
                        review_texts = []
                        if fsq_place_id:
                            try:
                                tips_url = f"https://places-api.foursquare.com/places/{fsq_place_id}/tips"
                                tips_res = requests.get(tips_url, headers=headers, timeout=5)
                                if tips_res.status_code == 200:
                                    tips_data = tips_res.json()
                                    tips = tips_data.get('results', []) if isinstance(tips_data, dict) else []
                                    review_texts = [tip.get("text", "") for tip in tips[:3]] if tips else []
                            except:
                                pass

                        # Generate different number of reviews for each restaurant (3-7 reviews)
                        num_reviews = random.randint(3, 7)
                        
                        # If no real reviews, generate AI ones with varying sentiment
                        if not review_texts:
                            # Generate a random rating between 3.0 and 5.0
                            base_rating = round(random.uniform(3.0, 5.0), 1)
                            # Use AI model to generate reviews
                            review_texts = generate_ai_reviews(name, food, base_rating, num_reviews)
                        
                        # Sentiment analysis
                        sentiments = []
                        for tip in review_texts:
                            try:
                                result = classifier(tip[:512])[0]
                                stars = int(result["label"].split()[0])
                                sentiments.append(stars)
                            except:
                                # If sentiment analysis fails, use a random rating based on review content
                                sentiment_score = random.randint(3, 5) if any(word in tip.lower() for word in ['great', 'amazing', 'excellent', 'love', 'best']) else random.randint(1, 3)
                                sentiments.append(sentiment_score)

                        # Get image from reliable fallback service (no Google API)
                        photo_url = get_restaurant_image(name, location, food)

                        avg_rating = round(sum(sentiments) / len(sentiments), 2) if sentiments else 0

                        results.append({
                            "Restaurant": name,
                            "Address": address,
                            "Google Maps Link": maps_link,
                            "Rating": avg_rating,
                            "Stars": "⭐" * int(round(avg_rating)) if avg_rating > 0 else "No reviews",
                            "Reviews": len(sentiments),
                            "Image": photo_url,
                            "Tips": review_texts,
                            "NumReviews": len(sentiments)
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
            tab1, tab2 = st.tabs(["Rating Distribution", "Review Insights"])
            
            with tab1:
                # Rating distribution chart
                fig = px.histogram(analysis_df, x='Rating', 
                                  title='Distribution of Restaurant Ratings',
                                  nbins=10, color='Rating',
                                  color_continuous_scale='viridis')
                st.plotly_chart(fig, use_container_width=True)
                
                # Show statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Average Rating", f"{analysis_df['Rating'].mean():.2f} ⭐")
                with col2:
                    st.metric("Total Reviews", f"{analysis_df['Reviews'].sum()}")
                with col3:
                    st.metric("Top Rated", f"{analysis_df['Rating'].max():.2f} ⭐")
            
            with tab2:
                # Sentiment analysis of reviews
                st.markdown("### 💬 AI-Generated Review Highlights")
                
                # Get all review texts
                all_reviews = [review for sublist in analysis_df['Tips'] for review in sublist]
                
                if all_reviews:
                    # Show word cloud of common terms
                    text = ' '.join(all_reviews)
                    wordcloud = WordCloud(width=800, height=400, background_color='black').generate(text)
                    
                    plt.figure(figsize=(10, 5))
                    plt.imshow(wordcloud, interpolation='bilinear')
                    plt.axis("off")
                    st.pyplot(plt)
                    
                    # Show sample reviews from top restaurants
                    st.markdown("### 📝 Sample Reviews from Top Restaurants")
                    top_restaurants = analysis_df.nlargest(3, 'Rating')
                    for idx, restaurant in top_restaurants.iterrows():
                        st.markdown(f"**{restaurant['Restaurant']}** ({restaurant['Rating']}⭐):")
                        for i, review in enumerate(restaurant['Tips'][:2], 1):
                            st.markdown(f"{i}. _{review}_")
                        st.markdown("---")
                else:
                    st.warning("No reviews available for analysis")
        else:
            st.info("No rating data available for analysis in current search results")

        # ======== TOP PICKS SECTION ========
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
                    # Display restaurant image
                    try:
                        st.image(r["Image"], use_column_width=True, caption=r["Restaurant"])
                    except:
                        st.image("https://images.unsplash.com/photo-1517248135467-4c7edcad34c4?w=400&h=300&fit=crop", 
                                use_column_width=True, caption="Restaurant Image")
                    
                    st.markdown(f"""
                        <div style="background-color: {color}; border-radius: 15px; padding: 20px; text-align: center; box-shadow: 0 4px 8px rgba(0,0,0,0.2); color: black; font-weight: bold;">
                            <div style="font-size: 22px; margin-bottom: 10px;">{medal}</div>
                            <div style="font-size: 18px; margin-bottom: 8px;">{r['Restaurant']}</div>
                            <div style="font-size: 15px; margin-bottom: 8px;">{r['Address']}</div>
                            <div style="font-size: 16px;">{r['Stars']} ({r['Rating']})</div>
                            <div style="font-size: 14px; margin-bottom: 8px;">{r['Reviews']} reviews</div>
                            <div style="margin-top: 10px;">
                                <a href="{r['Google Maps Link']}" target="_blank" class="map-link">📍 locate restaurant</a>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

        # Gallery Pick Section
        st.divider()
        st.subheader("🖼️ Restaurant Gallery")

        # Create columns for the gallery
        gallery_cols = st.columns(3)
        
        for idx, r in enumerate(st.session_state.results[:6]):  # Show first 6 restaurants
            with gallery_cols[idx % 3]:
                try:
                    st.image(r["Image"], use_column_width=True, caption=f"{r['Restaurant']} - {r['Rating']}⭐")
                except:
                    st.image("https://images.unsplash.com/photo-1517248135467-4c7edcad34c4?w=400&h=300&fit=crop", 
                            use_column_width=True, caption=f"{r['Restaurant']} - {r['Rating']}⭐")
                
                st.markdown(f"""
                    <div style="text-align: center;">
                        <strong>{r['Restaurant']}</strong><br>
                        {r['Stars']} ({r['Rating']})<br>
                        {r['Reviews']} reviews<br>
                        <a href="{r['Google Maps Link']}" target="_blank" class="map-link">📍 View on Map</a>
                    </div>
                """, unsafe_allow_html=True)

        st.divider()
        if reviewed_restaurants:
            top = max(reviewed_restaurants, key=lambda x: x["Rating"])
            st.metric(label="🏆 Top Pick", value=top["Restaurant"], delta=f"{top['Rating']} ⭐ ({top['Reviews']} reviews)")

            top_pick = {
                "Restaurant": top["Restaurant"],
                "Rating": top["Rating"],
                "Address": top["Address"],
                "Google Maps Link": top["Google Maps Link"],
                "Food": food,
                "Location": location,
                "ReviewCount": top["Reviews"]
            }
            append_history(top_pick)
        else:
            st.warning("No restaurants with reviews found to select a top pick.")

        st.divider()
        st.subheader("📝 Detailed Restaurant Reviews")

        for idx, r in enumerate(sorted(st.session_state.results, key=lambda x: x["Rating"] if x["Rating"] > 0 else 0, reverse=True)):
            with st.expander(f"{r['Restaurant']} - {r['Rating']}⭐ ({r['Reviews']} reviews)"):
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    try:
                        st.image(r["Image"], use_column_width=True, caption=r["Restaurant"])
                    except:
                        st.image("https://images.unsplash.com/photo-1517248135467-4c7edcad34c4?w=400&h=300&fit=crop", 
                                use_column_width=True, caption=r["Restaurant"])
                    
                    st.markdown(f"**📍 Address:** {r['Address']}")
                    st.markdown(f"**⭐ Rating:** {r['Rating']} ({r['Reviews']} reviews)")
                    st.markdown(f"[📍 View on Google Maps]({r['Google Maps Link']})", unsafe_allow_html=True)
                
                with col2:
                    st.markdown("### 💬 Customer Reviews")
                    for i, tip in enumerate(r["Tips"][:5], 1):
                        st.markdown(f"**Review {i}:** _{tip}_")
                    st.markdown(f"*AI-generated based on restaurant rating and cuisine type*")

# -------- PAGE: Deep Learning --------
elif st.session_state.page == "Deep Learning":
    st.title("🤖 Deep Learning Explained")
    st.markdown("""
    This app uses **BERT-based sentiment analysis** and **GPT-2 text generation** to evaluate and create restaurant reviews.

    ### How it works:
    - **Foursquare Places API**: Fetches nearby restaurants based on your food and location
    - **BERT Sentiment Analysis**: Analyzes review sentiment using Hugging Face transformers
    - **GPT-2 Text Generation**: Creates realistic reviews when real ones aren't available
    - **Multiple Image Services**: Uses reliable free image APIs (no Google API needed)
    - **AI-Powered Ranking**: Ranks restaurants based on AI-analyzed sentiment scores

    Each restaurant gets a different number of reviews (3-7) generated by local AI models!
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
    **AI Restaurant Recommender** is a Streamlit web app designed to help you discover top restaurants using:

    - **Foursquare Places API**: For restaurant data and locations
    - **Free Image Services**: Unsplash, Picsum, and Foodish APIs for restaurant images
    - **Hugging Face Transformers**: 
      - BERT for sentiment analysis
      - GPT-2 for AI-generated reviews
    - **Firebase Firestore**: For saving recommendation history
    - **Google Maps**: For navigation to recommended restaurants

    ### Features:
    - AI-generated reviews with varying counts per restaurant (3-7)
    - Sentiment-based rating system
    - High-quality restaurant images from free APIs
    - Interactive data visualizations
    - Historical recommendation tracking

    --- 
    _Powered by cutting-edge AI and reliable free APIs_
    """)

# Footer
st.markdown('<div class="custom-footer">© 2025 AI Restaurant Recommender</div>', unsafe_allow_html=True)
