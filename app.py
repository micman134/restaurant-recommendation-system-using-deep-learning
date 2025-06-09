import streamlit as st
import requests
import pandas as pd
try:
    from transformers import pipeline
except ImportError:
    st.error("Transformers library not installed correctly. Please install with: pip install transformers==4.30.0")
    st.stop()
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# Set page configuration
st.set_page_config(page_title="🍽️ Restaurant Recommender", layout="wide")

# Hide Streamlit UI and footer
st.markdown("""
    <style>
    #MainMenu, footer, header {visibility: hidden;}
    .stDeployButton {display: none;}
    .custom-footer {
        text-align: center;
        font-size: 14px;
        margin-top: 50px;
        padding: 20px;
        color: #aaa;
    }
    .restaurant-card {
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        background-color: #f9f9f9;
    }
    .zero-review {
        border-left: 5px solid #ff9800;
    }
    .top-rated {
        border-left: 5px solid #4caf50;
    }
    </style>
""", unsafe_allow_html=True)

# Load sentiment analysis model
@st.cache_resource(show_spinner=False)
def get_classifier():
    try:
        return pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
    except Exception as e:
        st.error(f"Failed to load sentiment analysis model: {str(e)}")
        st.stop()

# Google Sheets helpers
@st.cache_resource
def get_gsheet():
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_dict(st.secrets["gcp_service_account"], scope)
    client = gspread.authorize(creds)
    return client.open("Restaurant_Recommender_History").sheet1

def read_history():
    try:
        sheet = get_gsheet()
        return sheet.get_all_records()
    except Exception as e:
        st.error(f"Failed to read from Google Sheets: {str(e)}")
        return []

def append_history(data_dict):
    try:
        food = data_dict.get("Food", "").strip()
        location = data_dict.get("Location", "").strip()

        if not food or not location:
            return

        sheet = get_gsheet()
        existing_rows = sheet.get_all_records()

        # Check for duplicate entry
        for row in existing_rows:
            if (row.get("Restaurant") == data_dict.get("Restaurant") and
                row.get("Food") == food and
                row.get("Location") == location):
                return

        row = [
            data_dict.get("Restaurant", ""),
            data_dict.get("Rating", ""),
            data_dict.get("Address", ""),
            food,
            location
        ]
        sheet.append_row(row)
        st.success("New recommendation saved to history!")
    except Exception as e:
        st.error(f"Failed to save to history: {str(e)}")

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
        food = st.text_input("🍕 Food Type", placeholder="e.g., Sushi, Pizza, Burgers")

    col1, _ = st.columns([1, 1])
    with col1:
        location = st.text_input("📍 Location", placeholder="e.g., New York, London")

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
                try:
                    headers = {"accept": "application/json", "Authorization": api_key}
                    params = {"query": food, "near": location, "limit": 40}
                    res = requests.get("https://api.foursquare.com/v3/places/search", headers=headers, params=params)
                    restaurants = res.json().get("results", [])

                    if not restaurants:
                        st.error("❌ No restaurants found. Try different search terms.")
                    else:
                        classifier = get_classifier()
                        results = []
                        zero_review_restaurants = []

                        for r in restaurants:
                            try:
                                fsq_id = r['fsq_id']
                                name = r['name']
                                address = r['location'].get('formatted_address', 'Unknown')

                                tips_url = f"https://api.foursquare.com/v3/places/{fsq_id}/tips"
                                tips_res = requests.get(tips_url, headers=headers)
                                tips = tips_res.json()
                                review_texts = [tip["text"] for tip in tips[:5]] if isinstance(tips, list) else []

                                # Get photo
                                photo_url = ""
                                photo_api = f"https://api.foursquare.com/v3/places/{fsq_id}/photos"
                                photo_res = requests.get(photo_api, headers=headers)
                                photos = photo_res.json()
                                if photos and isinstance(photos, list):
                                    photo = photos[0]
                                    photo_url = f"{photo['prefix']}original{photo['suffix']}"

                                if review_texts:
                                    sentiments = []
                                    for tip in review_texts:
                                        result = classifier(tip[:512])[0]
                                        stars = int(result["label"].split()[0])
                                        sentiments.append(stars)

                                    avg_rating = round(sum(sentiments) / len(sentiments), 2) if sentiments else 0

                                    results.append({
                                        "Restaurant": name,
                                        "Address": address,
                                        "Rating": avg_rating,
                                        "Stars": "⭐" * int(round(avg_rating)),
                                        "Reviews": len(sentiments),
                                        "Image": photo_url,
                                        "Tips": review_texts[:2],
                                        "HasReviews": True
                                    })
                                else:
                                    zero_review_restaurants.append({
                                        "Restaurant": name,
                                        "Address": address,
                                        "Rating": 0,
                                        "Stars": "⭐" * 0,
                                        "Reviews": 0,
                                        "Image": photo_url,
                                        "Tips": [],
                                        "HasReviews": False
                                    })
                            except Exception as e:
                                st.error(f"Error processing restaurant {name}: {str(e)}")
                                continue

                        # Combine both lists for display
                        all_restaurants = results + zero_review_restaurants
                        
                        if all_restaurants:
                            df = pd.DataFrame([{
                                "Restaurant": r["Restaurant"],
                                "Address": r["Address"],
                                "Rating": r["Rating"],
                                "Stars": r["Stars"],
                                "Reviews": r["Reviews"],
                                "Has Reviews": "Yes" if r["HasReviews"] else "No"
                            } for r in all_restaurants])
                            df.index += 1
                            st.session_state.results = all_restaurants
                            st.session_state.df = df
                        else:
                            st.warning("No restaurants found with reviews.")
                except Exception as e:
                    st.error(f"An error occurred during search: {str(e)}")

    if st.session_state.results:
        st.divider()
        st.subheader("📊 All Restaurants")
        st.dataframe(st.session_state.df, use_container_width=True)

        # Filter out restaurants with reviews for top picks
        reviewed_restaurants = [r for r in st.session_state.results if r["HasReviews"]]
        
        if reviewed_restaurants:
            top3 = sorted(reviewed_restaurants, key=lambda x: x["Rating"], reverse=True)[:3]
            st.divider()
            st.subheader("🏅 Top Rated Restaurants")

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
                            </div>
                        """, unsafe_allow_html=True)

            st.divider()
            st.subheader("🌟 Restaurant Highlights")
            
            tab1, tab2, tab3 = st.tabs(["Top Rated", "New Discoveries", "All Restaurants"])
            
            with tab1:
                st.markdown("### 🏆 Top Rated Restaurants")
                top_rated = sorted(reviewed_restaurants, key=lambda x: x["Rating"], reverse=True)[:5]
                for r in top_rated:
                    with st.container():
                        st.markdown(f"""
                            <div class="restaurant-card top-rated">
                                <h3>{r['Restaurant']} <span style="color: #4caf50;">⭐ {r['Rating']}</span></h3>
                                <p><strong>📍 Address:</strong> {r['Address']}</p>
                                <p><strong>📊 Reviews:</strong> {r['Reviews']} reviews</p>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        if r["Image"]:
                            st.image(r["Image"], use_column_width=True)
                        
                        if r["Tips"]:
                            with st.expander("See recent reviews"):
                                for tip in r["Tips"]:
                                    st.markdown(f"- _{tip}_")
            
            with tab2:
                st.markdown("### 🆕 New Discoveries")
                if zero_review_restaurants:
                    for r in zero_review_restaurants[:5]:
                        with st.container():
                            st.markdown(f"""
                                <div class="restaurant-card zero-review">
                                    <h3>{r['Restaurant']}</h3>
                                    <p><strong>📍 Address:</strong> {r['Address']}</p>
                                    <p><em>No reviews yet - Be the first to try!</em></p>
                                </div>
                            """, unsafe_allow_html=True)
                            
                            if r["Image"]:
                                st.image(r["Image"], use_column_width=True)
                else:
                    st.info("No new restaurants without reviews found in this search.")
            
            with tab3:
                st.markdown("### 🍽️ All Restaurants")
                for r in st.session_state.results:
                    with st.container():
                        if r["HasReviews"]:
                            st.markdown(f"""
                                <div class="restaurant-card">
                                    <h3>{r['Restaurant']} <span style="color: #2196f3;">⭐ {r['Rating']}</span></h3>
                                    <p><strong>📍 Address:</strong> {r['Address']}</p>
                                    <p><strong>📊 Reviews:</strong> {r['Reviews']} reviews</p>
                                </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                                <div class="restaurant-card zero-review">
                                    <h3>{r['Restaurant']}</h3>
                                    <p><strong>📍 Address:</strong> {r['Address']}</p>
                                    <p><em>No reviews available</em></p>
                                </div>
                            """, unsafe_allow_html=True)
                        
                        if r["Image"]:
                            st.image(r["Image"], use_column_width=True)
                        
                        if r["Tips"]:
                            with st.expander("See recent reviews"):
                                for tip in r["Tips"]:
                                    st.markdown(f"- _{tip}_")

            if reviewed_restaurants:
                st.divider()
                top = max(reviewed_restaurants, key=lambda x: x["Rating"])
                st.metric(label="🏆 Top Pick", value=top["Restaurant"], delta=f"{top['Rating']} ⭐")

                top_pick = {
                    "Restaurant": top["Restaurant"],
                    "Rating": top["Rating"],
                    "Address": top["Address"],
                    "Food": food,
                    "Location": location
                }
                append_history(top_pick)

# -------- PAGE: Deep Learning --------
elif st.session_state.page == "Deep Learning":
    st.title("🤖 Deep Learning Explained")
    st.markdown("""
    This app uses **BERT-based sentiment analysis** to evaluate restaurant reviews and provide AI-driven recommendations.

    ### How it works:
    - Fetches nearby restaurants from the **Foursquare API** based on your food and location input
    - Retrieves recent user reviews ("tips") for each restaurant
    - Uses a pretrained **BERT sentiment analysis model** to analyze the sentiment
    - Calculates an average rating score from the sentiment predictions
    - Ranks restaurants by these AI-driven scores to recommend the best places

    The model used is `nlptown/bert-base-multilingual-uncased-sentiment` which is specifically fine-tuned for sentiment analysis.
    """)

# -------- PAGE: History --------
elif st.session_state.page == "History":
    st.title("📚 Recommendation History")
    try:
        history_data = read_history()
        if not history_data:
            st.info("No history available yet. Try making some recommendations first!")
        else:
            df_hist = pd.DataFrame(history_data)
            df_hist.index += 1
            st.dataframe(df_hist, use_container_width=True)
    except Exception as e:
        st.error(f"Failed to load history: {str(e)}")

# -------- PAGE: About --------
elif st.session_state.page == "About":
    st.title("ℹ️ About This App")
    st.markdown("""
    **AI Restaurant Recommender** is a Streamlit web app that helps you discover the best restaurants based on AI analysis of real user reviews.

    ### Features:
    - Finds restaurants near any location worldwide
    - Analyzes review sentiment using state-of-the-art NLP
    - Highlights both top-rated and new establishments
    - Saves your search history

    ### Technologies Used:
    - [Foursquare API](https://developer.foursquare.com/) for restaurant data
    - Hugging Face Transformers for sentiment analysis
    - Google Sheets for history storage
    - Streamlit for the web interface

    --- 
    *Created with ❤️ using Python and AI*
    """)

# Footer
st.markdown('<div class="custom-footer">© 2023 AI Restaurant Recommender</div>', unsafe_allow_html=True)
