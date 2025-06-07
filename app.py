import streamlit as st
import requests
import pandas as pd
from transformers import pipeline
import matplotlib.pyplot as plt
import io
import base64
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# Set page config
st.set_page_config(page_title="🍽️ Restaurant Recommender", layout="wide")

# Hide Streamlit UI and footer
st.markdown("""
    <style>
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
    </style>
""", unsafe_allow_html=True)

# Load sentiment classifier once
@st.cache_resource(show_spinner=False)
def get_classifier():
    return pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

# Google Sheets helpers
@st.cache_resource
def get_gsheet():
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_dict(st.secrets["gcp_service_account"], scope)
    client = gspread.authorize(creds)
    sheet = client.open("Restaurant_Recommender_History").sheet1
    return sheet

def read_history():
    sheet = get_gsheet()
    return sheet.get_all_records()

def append_history(data_dict):
    sheet = get_gsheet()
    sheet.append_row(list(data_dict.values()))

# Function to create small bar chart as base64 PNG
def create_sentiment_chart(positive, neutral, negative):
    fig, ax = plt.subplots(figsize=(2, 1.2))
    bars = ax.bar(['Positive', 'Neutral', 'Negative'], [positive, neutral, negative],
                  color=['#4CAF50', '#FFC107', '#F44336'])
    ax.set_ylim(0, 100)
    ax.set_ylabel('%')
    ax.set_title('Sentiments', fontsize=10)
    ax.tick_params(axis='x', labelrotation=0, labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=80)
    plt.close(fig)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode()
    return f"data:image/png;base64,{img_base64}"

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
                params = {"query": food, "near": location, "limit": 10}
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

                        # Fetch tips/reviews
                        tips_url = f"https://api.foursquare.com/v3/places/{fsq_id}/tips"
                        tips_res = requests.get(tips_url, headers=headers)
                        tips = tips_res.json()
                        review_texts = [tip["text"] for tip in tips[:5]] if tips else []

                        sentiments = []
                        sentiment_counts = {"Positive": 0, "Neutral": 0, "Negative": 0}
                        for tip in review_texts:
                            result = classifier(tip[:512])[0]
                            stars = int(result["label"].split()[0])
                            sentiments.append(stars)
                            # Categorize sentiment by star rating
                            if stars >= 4:
                                sentiment_counts["Positive"] += 1
                            elif stars == 3:
                                sentiment_counts["Neutral"] += 1
                            else:
                                sentiment_counts["Negative"] += 1

                        total_reviews = len(sentiments)
                        avg_rating = round(sum(sentiments) / total_reviews, 2) if total_reviews else 0

                        # Calculate percentages
                        if total_reviews > 0:
                            positive_pct = round((sentiment_counts["Positive"] / total_reviews) * 100, 1)
                            neutral_pct = round((sentiment_counts["Neutral"] / total_reviews) * 100, 1)
                            negative_pct = round((sentiment_counts["Negative"] / total_reviews) * 100, 1)
                        else:
                            positive_pct = neutral_pct = negative_pct = 0

                        # Fetch photo
                        photo_url = ""
                        photo_api = f"https://api.foursquare.com/v3/places/{fsq_id}/photos"
                        photo_res = requests.get(photo_api, headers=headers)
                        photos = photo_res.json()
                        if photos:
                            photo = photos[0]
                            photo_url = f"{photo['prefix']}original{photo['suffix']}"

                        if sentiments:
                            results.append({
                                "Restaurant": name,
                                "Address": address,
                                "Rating": avg_rating,
                                "Stars": "⭐" * int(round(avg_rating)),
                                "Reviews": total_reviews,
                                "Image": photo_url,
                                "Tips": review_texts[:2],
                                "Positive %": positive_pct,
                                "Neutral %": neutral_pct,
                                "Negative %": negative_pct
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
                        st.warning("Found restaurants, but no reviews available.")

    if st.session_state.results:
        st.divider()
        st.subheader("📊 Restaurants Search Results and Ratings")
        st.dataframe(st.session_state.df, use_container_width=True)

        top3 = sorted(st.session_state.results, key=lambda x: x["Rating"], reverse=True)[:3]
        st.divider()
        st.subheader("🏅 AI (Deep Learning) Top Picks")

        cols = st.columns(3)
        medals = ["🥇 1st", "🥈 2nd", "🥉 3rd"]
        colors = ["#FFD700", "#C0C0C0", "#CD7F32"]

        for i, (col, medal, color) in enumerate(zip(cols, medals, colors)):
            if i < len(top3):
                r = top3[i]
                with col:
                    # Generate small sentiment chart
                    chart_uri = create_sentiment_chart(r.get("Positive %", 0), r.get("Neutral %", 0), r.get("Negative %", 0))

                    st.markdown(f"""
                        <div style="background-color: {color}; border-radius: 15px; padding: 15px; text-align: center; box-shadow: 0 4px 8px rgba(0,0,0,0.2); color: black; font-weight: bold;">
                            <div style="font-size: 22px; margin-bottom: 10px;">{medal}</div>
                            <div style="font-size: 18px; margin-bottom: 8px;">{r['Restaurant']}</div>
                            <div style="font-size: 15px; margin-bottom: 8px;">{r['Address']}</div>
                            <div style="font-size: 16px;">{r['Stars']} ({r['Rating']})</div>
                            <img src="{chart_uri}" style="margin-top: 10px; margin-bottom: 10px; width: 100%;">
                            <div style="font-size: 14px; text-align: left;">
                                <b>Reviews:</b> {r['Reviews']}<br>
                                <b>Sample Tips:</b> <i>{'<br>'.join(r['Tips'])}</i>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

# -------- PAGE: History --------
elif st.session_state.page == "History":
    st.title("📜 Search History")

    try:
        records = read_history()
        if not records:
            st.info("No history found yet.")
        else:
            history_df = pd.DataFrame(records)
            st.dataframe(history_df, use_container_width=True)
    except Exception as e:
        st.error(f"Error loading history: {e}")

# -------- PAGE: Deep Learning --------
elif st.session_state.page == "Deep Learning":
    st.title("🤖 About AI & Sentiment Analysis")
    st.markdown("""
    This app uses **BERT-based sentiment analysis** to evaluate restaurant reviews fetched from **Foursquare** API.
    
    The classifier is a fine-tuned multilingual BERT model (nlptown/bert-base-multilingual-uncased-sentiment) that predicts star ratings (1 to 5).
    
    Sentiment scores are used to help rank restaurants based on positive, neutral, and negative review percentages.
    """)

# -------- PAGE: About --------
elif st.session_state.page == "About":
    st.title("ℹ️ About This App")
    st.markdown("""
    - Developed using Streamlit, Foursquare API, and HuggingFace Transformers.
    - Sentiment analysis uses a pretrained BERT model.
    - Created by your assistant 💡
    """)

# Footer
st.markdown('<div class="custom-footer">Made with ❤️ using Streamlit & Python</div>', unsafe_allow_html=True)
