import streamlit as st
import requests
import pandas as pd
from transformers import pipeline
from datetime import datetime
import urllib.parse
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import json
import random
import time
import re

# Try to import BeautifulSoup with error handling
try:
    from bs4 import BeautifulSoup
    BEAUTIFULSOUP_AVAILABLE = True
except ImportError:
    BEAUTIFULSOUP_AVAILABLE = False
    st.warning("BeautifulSoup not installed. Web scraping features will be limited.")

# Try to import Firebase with error handling
try:
    import firebase_admin
    from firebase_admin import credentials, firestore
    FIREBASE_AVAILABLE = True
except ImportError:
    FIREBASE_AVAILABLE = False

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
    
    .scraping-notice {
        background-color: #ff9800;
        color: white;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
        border-left: 4px solid #f57c00;
    }
    
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
        color: #856404;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load sentiment analysis model with error handling
@st.cache_resource(show_spinner=False)
def get_classifier():
    try:
        return pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
    except Exception as e:
        st.error(f"Failed to load sentiment analysis model: {e}")
        return None

# Mock restaurant data
def get_mock_restaurants(food_type, location):
    mock_restaurants = [
        {
            "name": f"Delicious {food_type.title()} House",
            "address": f"123 Main St, {location}",
            "rating": round(random.uniform(3.5, 5.0), 1),
            "reviews": random.randint(5, 50),
            "image": "https://images.unsplash.com/photo-1517248135467-4c7edcad34c4",
            "tips": [
                f"Amazing {food_type} here! Highly recommended.",
                f"Great atmosphere and friendly staff. The {food_type} was delicious."
            ]
        },
        {
            "name": f"{location.title()} {food_type.title()} Palace",
            "address": f"456 Oak Ave, {location}",
            "rating": round(random.uniform(3.0, 4.8), 1),
            "reviews": random.randint(3, 30),
            "image": "https://images.unsplash.com/photo-1555396273-367ea4eb4db5",
            "tips": [
                f"Authentic {food_type} experience. Will come back again!",
                f"Good quality {food_type} at reasonable prices."
            ]
        },
        {
            "name": f"{food_type.title()} Express",
            "address": f"789 Pine Rd, {location}",
            "rating": round(random.uniform(4.0, 4.9), 1),
            "reviews": random.randint(10, 40),
            "image": "https://images.unsplash.com/photo-1590846406792-0adc7f938f1d",
            "tips": [
                f"Quick service and tasty {food_type}. Perfect for lunch!",
                f"Love their specialty {food_type}. Always fresh ingredients."
            ]
        },
        {
            "name": f"Golden {food_type.title()} Restaurant",
            "address": f"321 Elm St, {location}",
            "rating": round(random.uniform(3.2, 4.5), 1),
            "reviews": random.randint(8, 25),
            "image": "https://images.unsplash.com/photo-1578474846511-04ba529f0b88",
            "tips": [
                f"Cozy place with great {food_type}. Good for families.",
                f"Traditional {food_type} recipe. Very authentic taste."
            ]
        },
        {
            "name": f"{food_type.title()} Garden",
            "address": f"654 Maple Dr, {location}",
            "rating": round(random.uniform(4.2, 4.7), 1),
            "reviews": random.randint(15, 35),
            "image": "https://images.unsplash.com/photo-1467003909585-2f8a72700288",
            "tips": [
                f"Beautiful ambiance and excellent {food_type}. Romantic atmosphere.",
                f"Creative {food_type} dishes. Great presentation and taste."
            ]
        }
    ]
    return mock_restaurants

def get_headers():
    return {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Connection': 'keep-alive',
    }

def simple_web_search(food_type, location):
    """Simple web search without BeautifulSoup"""
    try:
        # This is a very basic approach that doesn't require HTML parsing
        search_query = f"{food_type} restaurants in {location}"
        encoded_query = urllib.parse.quote_plus(search_query)
        
        # Just return mock data based on search query
        mock_restaurants = get_mock_restaurants(food_type, location)
        
        # Add source information
        for restaurant in mock_restaurants:
            restaurant['source'] = 'Web Search (Mock)'
        
        return mock_restaurants
        
    except Exception as e:
        st.error(f"Web search failed: {e}")
        return []

def get_mock_reviews(restaurant_name, food_type):
    """Generate realistic mock reviews"""
    review_templates = [
        f"Amazing {food_type} at {restaurant_name}! The quality was outstanding and service was excellent.",
        f"Great atmosphere and friendly staff. The {food_type} was cooked to perfection.",
        f"Highly recommend {restaurant_name} for their delicious {food_type}. Will definitely return!",
        f"Good {food_type} but service could be better. Overall a decent experience.",
        f"Authentic {food_type} experience. The flavors were incredible and prices reasonable.",
        f"Quick service and tasty {food_type}. Perfect for a quick lunch or dinner.",
        f"Cozy place with great {food_type}. Good for families and casual dining.",
        f"Traditional {food_type} recipe. Very authentic taste and presentation.",
        f"Beautiful ambiance and excellent {food_type}. Romantic atmosphere for dates.",
        f"Creative {food_type} dishes. Great presentation and taste combination."
    ]
    
    return random.sample(review_templates, min(3, len(review_templates)))

# Session state initialization
if "page" not in st.session_state:
    st.session_state.page = "Recommend"
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
    if st.button("About"):
        st.session_state.page = "About"
    
    st.divider()
    st.markdown("### ⚙️ Settings")
    
    if BEAUTIFULSOUP_AVAILABLE:
        scraping_source = st.selectbox(
            "Data Source",
            ["Demo Mode", "Simple Web Search"],
            index=0,
            help="Select data source method"
        )
    else:
        st.info("🔧 Using Demo Mode (install BeautifulSoup for web scraping)")
        scraping_source = "Demo Mode"

# -------- PAGE: Recommend --------
if st.session_state.page == "Recommend":
    st.title("🍽️ AI Restaurant Recommender")
    st.markdown("Find top-rated restaurants using **AI sentiment analysis**")
    
    # Installation instructions if BeautifulSoup is missing
    if not BEAUTIFULSOUP_AVAILABLE:
        st.markdown("""
        <div class="warning-box">
        ⚠️ <strong>Package Missing:</strong> BeautifulSoup is not installed. <br>
        <strong>To install:</strong> Run <code>pip install beautifulsoup4</code> in your terminal
        </div>
        """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1])
    with col1:
        food = st.text_input("🍕 Food Type", placeholder="e.g., Sushi, Pizza, Burger")
    with col2:
        location = st.text_input("📍 Location", placeholder="e.g., New York, London")

    if st.button("🔍 Search Restaurants"):
        if not food or not location:
            st.warning("⚠️ Please enter both a food type and location.")
        else:
            with st.spinner(f"Searching for {food} in {location}..."):
                time.sleep(1)  # Simulate search delay
                
                restaurants = []
                
                if scraping_source == "Simple Web Search" and BEAUTIFULSOUP_AVAILABLE:
                    restaurants = simple_web_search(food, location)
                else:
                    # Use demo mode
                    restaurants = get_mock_restaurants(food, location)
                    for restaurant in restaurants:
                        restaurant['source'] = 'Demo Mode'
                
                if not restaurants:
                    st.warning("No restaurants found. Using demo data instead.")
                    restaurants = get_mock_restaurants(food, location)
                    for restaurant in restaurants:
                        restaurant['source'] = 'Demo Fallback'

                # Process results
                results = []
                classifier = get_classifier()
                
                for restaurant in restaurants:
                    maps_query = urllib.parse.quote_plus(f"{restaurant['name']}, {restaurant['address']}")
                    maps_link = f"https://www.google.com/maps/search/?api=1&query={maps_query}"
                    
                    # Generate mock reviews and analyze sentiment
                    reviews = get_mock_reviews(restaurant['name'], food)
                    sentiments = []
                    
                    if classifier:
                        for review in reviews:
                            try:
                                result = classifier(review[:512])[0]
                                stars = int(result["label"].split()[0])
                                sentiments.append(stars)
                            except:
                                sentiments.append(random.randint(3, 5))
                    else:
                        sentiments = [random.randint(3, 5) for _ in reviews]
                    
                    avg_rating = restaurant['rating']
                    if sentiments:
                        avg_rating = round(sum(sentiments) / len(sentiments), 1)
                    
                    results.append({
                        "Restaurant": restaurant['name'],
                        "Address": restaurant['address'],
                        "Google Maps Link": maps_link,
                        "Rating": avg_rating,
                        "Stars": "⭐" * int(round(avg_rating)),
                        "Reviews": restaurant['reviews'],
                        "Image": restaurant['image'],
                        "Tips": reviews,
                        "Source": restaurant.get('source', 'Demo Mode')
                    })
                
                st.session_state.results = results
                
                # Create DataFrame
                df = pd.DataFrame([{
                    "Restaurant": r["Restaurant"],
                    "Address": r["Address"],
                    "Average Rating": r["Rating"],
                    "Stars": r["Stars"],
                    "Reviews": r["Reviews"],
                    "Source": r["Source"]
                } for r in results])
                df.index += 1
                st.session_state.df = df

    # Display results if available
    if st.session_state.results:
        st.divider()
        st.subheader("📊 Restaurant Recommendations")
        st.dataframe(st.session_state.df, use_container_width=True)
        
        # Show data source
        st.info(f"📡 Data source: {st.session_state.results[0]['Source']}")

        # Analysis Section
        st.divider()
        st.subheader("📈 Recommendation Analysis")
        
        analysis_df = pd.DataFrame(st.session_state.results)
        
        if analysis_df['Rating'].sum() > 0:
            tab1, tab2 = st.tabs(["Rating Distribution", "Review Insights"])
            
            with tab1:
                fig = px.histogram(analysis_df, x='Rating', title='Restaurant Rating Distribution',
                                  nbins=10, color_discrete_sequence=['#4CAF50'])
                st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                st.markdown("### 💬 Review Highlights")
                all_reviews = [review for sublist in analysis_df['Tips'] for review in sublist]
                
                if all_reviews:
                    text = ' '.join(all_reviews)
                    wordcloud = WordCloud(width=800, height=400, background_color='black').generate(text)
                    
                    plt.figure(figsize=(10, 5))
                    plt.imshow(wordcloud, interpolation='bilinear')
                    plt.axis("off")
                    st.pyplot(plt)
                    
                    st.markdown("### 📝 Sample Reviews")
                    for i, r in enumerate(analysis_df.to_dict('records')[:3], 1):
                        st.markdown(f"**{r['Restaurant']}** ({r['Rating']} ⭐)")
                        for tip in r['Tips'][:2]:
                            st.markdown(f"• _{tip}_")
                        st.markdown("---")

        # Top Picks Section
        st.divider()
        st.subheader("🏅 AI Top Picks")

        cols = st.columns(3)
        medals = ["🥇 1st", "🥈 2nd", "🥉 3rd"]
        colors = ["#FFD700", "#C0C0C0", "#CD7F32"]

        top3 = sorted(st.session_state.results, key=lambda x: x["Rating"], reverse=True)[:3]
        
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
        st.subheader("🖼️ Restaurant Gallery")

        gallery_cols = st.columns(3)
        for idx, r in enumerate(sorted(st.session_state.results, key=lambda x: x["Rating"], reverse=True)):
            if idx < 3:
                with gallery_cols[idx % 3]:
                    st.markdown(f"""
                        <div class="gallery-img-container">
                            <img src="{r['Image']}" class="gallery-img" />
                        </div>
                        <div class="gallery-caption">
                            <strong>{r['Restaurant']}</strong><br>
                            {r['Stars']} ({r['Rating']})<br>
                            <a href="{r['Google Maps Link']}" target="_blank" class="map-link">📍 View on Map</a>
                        </div>
                    """, unsafe_allow_html=True)

        # Top Pick Metric
        st.divider()
        top = max(st.session_state.results, key=lambda x: x["Rating"])
        st.metric(label="🏆 Top Pick", value=top["Restaurant"], delta=f"{top['Rating']} ⭐")

# -------- PAGE: Deep Learning --------
elif st.session_state.page == "Deep Learning":
    st.title("🤖 Deep Learning & AI Analysis")
    st.markdown("""
    This app uses **BERT-based sentiment analysis** to provide restaurant recommendations.

    ### How it works:
    - **Data Collection**: Restaurant information and reviews
    - **Sentiment Analysis**: BERT model analyzes review sentiment
    - **Rating Calculation**: Converts sentiment scores to star ratings
    - **Smart Ranking**: AI-driven restaurant recommendations

    ### Technical Stack:
    - **Hugging Face Transformers**: BERT model for sentiment analysis
    - **Streamlit**: Web application framework
    - **Plotly**: Interactive data visualizations
    - **Pandas**: Data processing and analysis

    ### Installation Requirements:
    ```bash
    pip install streamlit transformers pandas plotly wordcloud matplotlib
    ```
    
    For web scraping features:
    ```bash
    pip install beautifulsoup4 requests
    ```
    """)

# -------- PAGE: About --------
elif st.session_state.page == "About":
    st.title("ℹ️ About This App")
    st.markdown("""
    **AI Restaurant Recommender** - A smart restaurant discovery system

    ### Features:
    - **AI-Powered Analysis**: BERT sentiment analysis of reviews
    - **Smart Recommendations**: AI-driven restaurant rankings
    - **Beautiful UI**: Interactive and user-friendly interface
    - **Multiple Data Sources**: Flexible data collection methods

    ### Technology:
    - Built with **Streamlit** for the web interface
    - Uses **Hugging Face Transformers** for AI analysis
    - **Plotly** for interactive visualizations
    - **Pandas** for data processing

    ### Installation:
    ```bash
    # Core requirements
    pip install streamlit transformers pandas plotly wordcloud matplotlib
    
    # Optional: Web scraping
    pip install beautifulsoup4 requests
    
    # Run the app
    streamlit run app.py
    ```

    --- 
    _Built with Streamlit and Hugging Face Transformers_
    """)

# Footer
st.markdown("""
<div class="custom-footer">
© 2025 AI Restaurant Recommender | Educational Demo<br>
<small>AI-powered restaurant recommendation system</small>
</div>
""", unsafe_allow_html=True)
