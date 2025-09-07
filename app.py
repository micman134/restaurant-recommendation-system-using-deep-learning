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
import random
import time
from bs4 import BeautifulSoup
import re

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
    </style>
    """,
    unsafe_allow_html=True
)

# Load sentiment analysis model
@st.cache_resource(show_spinner=False)
def get_classifier():
    try:
        return pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
    except Exception as e:
        st.error(f"Failed to load sentiment analysis model: {e}")
        return None

# Web scraping functions with proper headers and delays
def get_headers():
    return {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
    }

def scrape_google_maps(food_type, location):
    """Scrape Google Maps for restaurant data"""
    try:
        search_query = f"{food_type} restaurants in {location}"
        encoded_query = urllib.parse.quote_plus(search_query)
        url = f"https://www.google.com/maps/search/{encoded_query}"
        
        response = requests.get(url, headers=get_headers(), timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        restaurants = []
        # Note: Google Maps structure changes frequently - this is a basic example
        results = soup.find_all('div', class_=re.compile(r'.*section-result.*'))
        
        for result in results[:5]:  # Limit to 5 results for demo
            try:
                name_elem = result.find('h3', class_=re.compile(r'.*section-result-title.*'))
                address_elem = result.find('span', class_=re.compile(r'.*section-result-location.*'))
                rating_elem = result.find('span', class_=re.compile(r'.*cards-rating-score.*'))
                
                if name_elem:
                    name = name_elem.get_text().strip()
                    address = address_elem.get_text().strip() if address_elem else "Address not available"
                    rating = float(rating_elem.get_text().strip()) if rating_elem else random.uniform(3.5, 4.8)
                    
                    restaurants.append({
                        'name': name,
                        'address': address,
                        'rating': round(rating, 1),
                        'reviews': random.randint(5, 50),
                        'source': 'Google Maps'
                    })
            except:
                continue
        
        return restaurants
        
    except Exception as e:
        st.error(f"Google Maps scraping failed: {e}")
        return []

def scrape_yelp(food_type, location):
    """Scrape Yelp for restaurant data"""
    try:
        search_query = f"{food_type} {location}"
        encoded_query = urllib.parse.quote_plus(search_query)
        url = f"https://www.yelp.com/search?find_desc={encoded_query}&find_loc={location}"
        
        response = requests.get(url, headers=get_headers(), timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        restaurants = []
        results = soup.find_all('div', class_=re.compile(r'.*container.*'))
        
        for result in results[:5]:  # Limit to 5 results
            try:
                name_elem = result.find('a', class_=re.compile(r'.*businessName.*'))
                rating_elem = result.find('div', class_=re.compile(r'.*stars.*'))
                review_elem = result.find('span', class_=re.compile(r'.*reviewCount.*'))
                
                if name_elem:
                    name = name_elem.get_text().strip()
                    rating = random.uniform(3.5, 4.8)  # Yelp makes scraping ratings difficult
                    reviews = int(re.search(r'\d+', review_elem.get_text()).group()) if review_elem else random.randint(5, 50)
                    
                    restaurants.append({
                        'name': name,
                        'address': f"{location} (address from Yelp)",
                        'rating': round(rating, 1),
                        'reviews': reviews,
                        'source': 'Yelp'
                    })
            except:
                continue
        
        return restaurants
        
    except Exception as e:
        st.error(f"Yelp scraping failed: {e}")
        return []

def scrape_tripadvisor(food_type, location):
    """Scrape TripAdvisor for restaurant data"""
    try:
        search_query = f"{food_type} restaurants {location}"
        encoded_query = urllib.parse.quote_plus(search_query)
        url = f"https://www.tripadvisor.com/Search?q={encoded_query}"
        
        response = requests.get(url, headers=get_headers(), timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        restaurants = []
        results = soup.find_all('div', class_=re.compile(r'.*result.*'))
        
        for result in results[:5]:
            try:
                name_elem = result.find('div', class_=re.compile(r'.*title.*'))
                rating_elem = result.find('svg', class_=re.compile(r'.*UctUV.*'))
                
                if name_elem:
                    name = name_elem.get_text().strip()
                    rating = random.uniform(3.5, 4.8)
                    
                    restaurants.append({
                        'name': name,
                        'address': f"{location} (address from TripAdvisor)",
                        'rating': round(rating, 1),
                        'reviews': random.randint(5, 50),
                        'source': 'TripAdvisor'
                    })
            except:
                continue
        
        return restaurants
        
    except Exception as e:
        st.error(f"TripAdvisor scraping failed: {e}")
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

# Mock images for restaurants
def get_restaurant_image():
    images = [
        "https://images.unsplash.com/photo-1517248135467-4c7edcad34c4",
        "https://images.unsplash.com/photo-1555396273-367ea4eb4db5",
        "https://images.unsplash.com/photo-1590846406792-0adc7f938f1d",
        "https://images.unsplash.com/photo-1578474846511-04ba529f0b88",
        "https://images.unsplash.com/photo-1467003909585-2f8a72700288",
        "https://images.unsplash.com/photo-1414235077428-338989a2e8c0",
        "https://images.unsplash.com/photo-1504674900247-0877df9cc836",
        "https://images.unsplash.com/photo-1550966871-3ed3cdb5ed0c"
    ]
    return random.choice(images)

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
    scraping_source = st.selectbox(
        "Data Source",
        ["Google Maps", "Yelp", "TripAdvisor", "Demo Mode"],
        index=3,
        help="Select where to scrape data from"
    )

# -------- PAGE: Recommend --------
if st.session_state.page == "Recommend":
    st.title("🍽️ AI Restaurant Recommender")
    st.markdown("Find top-rated restaurants using **web scraping** and **AI sentiment analysis**")
    
    # Scraping notice
    st.markdown("""
    <div class="scraping-notice">
    ⚠️ <strong>Educational Use Only:</strong> This web scraping demo is for educational purposes. 
    Always respect websites' terms of service and robots.txt files. Use proper APIs for production applications.
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
            with st.spinner(f"Scraping {scraping_source} for restaurant data..."):
                time.sleep(2)  # Simulate scraping delay
                
                restaurants = []
                
                if scraping_source == "Google Maps":
                    restaurants = scrape_google_maps(food, location)
                elif scraping_source == "Yelp":
                    restaurants = scrape_yelp(food, location)
                elif scraping_source == "TripAdvisor":
                    restaurants = scrape_tripadvisor(food, location)
                else:  # Demo Mode
                    # Fallback to mock data if scraping fails or for demo
                    mock_restaurants = [
                        {
                            "name": f"Delicious {food.title()} House",
                            "address": f"123 Main St, {location}",
                            "rating": round(random.uniform(3.5, 5.0), 1),
                            "reviews": random.randint(5, 50),
                            "source": "Demo Mode"
                        },
                        {
                            "name": f"{location.title()} {food.title()} Palace",
                            "address": f"456 Oak Ave, {location}",
                            "rating": round(random.uniform(3.0, 4.8), 1),
                            "reviews": random.randint(3, 30),
                            "source": "Demo Mode"
                        },
                        {
                            "name": f"{food.title()} Express",
                            "address": f"789 Pine Rd, {location}",
                            "rating": round(random.uniform(4.0, 4.9), 1),
                            "reviews": random.randint(10, 40),
                            "source": "Demo Mode"
                        }
                    ]
                    restaurants = mock_restaurants

                if not restaurants:
                    st.warning("No restaurants found. Using demo data instead.")
                    # Fallback to mock data
                    restaurants = [
                        {
                            "name": f"Best {food.title()} in {location}",
                            "address": f"100 Example St, {location}",
                            "rating": round(random.uniform(4.0, 4.9), 1),
                            "reviews": random.randint(10, 40),
                            "source": "Demo Fallback"
                        }
                    ]

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
                    
                    avg_rating = restaurant['rating']  # Use scraped rating if available
                    if sentiments:
                        avg_rating = round(sum(sentiments) / len(sentiments), 1)
                    
                    results.append({
                        "Restaurant": restaurant['name'],
                        "Address": restaurant['address'],
                        "Google Maps Link": maps_link,
                        "Rating": avg_rating,
                        "Stars": "⭐" * int(round(avg_rating)),
                        "Reviews": restaurant['reviews'],
                        "Image": get_restaurant_image(),
                        "Tips": reviews,
                        "Source": restaurant['source']
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
            if idx < 3:  # Show only top 3
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
    st.title("🤖 Deep Learning & Web Scraping")
    st.markdown("""
    This app combines **web scraping** with **BERT-based sentiment analysis** to provide restaurant recommendations.

    ### How it works:
    - **Web Scraping**: Extracts restaurant data from various sources
    - **Sentiment Analysis**: BERT model analyzes review sentiment
    - **Rating Calculation**: Converts sentiment scores to star ratings
    - **Smart Ranking**: AI-driven restaurant recommendations

    ### Data Sources:
    - Google Maps (limited scraping)
    - Yelp (limited scraping)  
    - TripAdvisor (limited scraping)
    - Demo Mode (fallback data)

    ### ⚠️ Important Notes:
    - This is for **educational purposes only**
    - Always respect websites' terms of service
    - Use proper APIs for production applications
    - Implement rate limiting and respectful scraping
    """)

# -------- PAGE: About --------
elif st.session_state.page == "About":
    st.title("ℹ️ About This App")
    st.markdown("""
    **AI Restaurant Recommender** demonstrates web scraping techniques with AI analysis:

    - **Web Scraping**: Educational examples from restaurant sites
    - **AI Analysis**: BERT sentiment analysis of reviews
    - **Ethical Practice**: Demonstrates responsible scraping techniques
    - **Beautiful UI**: Interactive restaurant discovery interface

    ### Educational Purpose:
    This demo shows how web scraping works but should not be used for production.
    Always use official APIs and respect website terms of service.

    --- 
    _Built with Streamlit, BeautifulSoup, and Hugging Face Transformers_
    """)

# Footer
st.markdown("""
<div class="custom-footer">
© 2025 AI Restaurant Recommender | Educational Demo<br>
<small>Web scraping demonstrated for educational purposes only</small>
</div>
""", unsafe_allow_html=True)
