import streamlit as st
import requests
import pandas as pd
from transformers import pipeline
import matplotlib.pyplot as plt
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# Set page configuration
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
    if not data_dict.get("Food") or not data_dict.get("Location"):
        return
    sheet = get_gsheet()
    records = sheet.get_all_records()
    # Avoid duplicate entries by matching Restaurant + Food + Location
    for rec in records:
        if (rec.get("Restaurant") == data_dict.get("Restaurant") and
            rec.get("Food") == data_dict.get("Food") and
            rec.get("Location") == data_dict.get("Location")):
            return  # duplicate found; do not append
    sheet.append_row(list(data_dict.values()))

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
            st.warning("\u26a0\ufe0f Please enter both a food type and location.")
        elif not api_key:
            st.error("\u274c Foursquare API key is missing.")
        else:
            st.session_state.results = None
            st.session_state.df = None

            with st.spinner("Searching and analyzing reviews..."):
                headers = {"accept": "application/json", "Authorization": api_key}
                params = {"query": food, "near": location, "limit": 10}
                res = requests.get("https://api.foursquare.com/v3/places/search", headers=headers, params=params)
                restaurants = res.json().get("results", [])

                if not restaurants:
                    st.error("\u274c No restaurants found. Try different search terms.")
                else:
                    classifier = get_classifier()
                    results = []

                    for r in restaurants:
                        fsq_id = r['fsq_id']
                        name = r['name']
                        address = r['location'].get('formatted_address', 'Unknown')

                        tips_url = f"https://api.foursquare.com/v3/places/{fsq_id}/tips"
                        tips_res = requests.get(tips_url, headers=headers)
                        tips = tips_res.json()
                        review_texts = [tip["text"] for tip in tips[:5]] if tips else []

                        sentiments = []
                        # Counters for sentiment breakdown
                        positive_count = 0
                        neutral_count = 0
                        negative_count = 0

                        for tip in review_texts:
                            result = classifier(tip[:512])[0]
                            stars = int(result["label"].split()[0])
                            sentiments.append(stars)

                            # Assume: 4-5 stars = positive, 3 = neutral, 1-2 = negative
                            if stars >= 4:
                                positive_count += 1
                            elif stars == 3:
                                neutral_count += 1
                            else:
                                negative_count += 1

                        photo_url = ""
                        photo_api = f"https://api.foursquare.com/v3/places/{fsq_id}/photos"
                        photo_res = requests.get(photo_api, headers=headers)
                        photos = photo_res.json()
                        if photos:
                            photo = photos[0]
                            photo_url = f"{photo['prefix']}original{photo['suffix']}"

                        avg_rating = round(sum(sentiments) / len(sentiments), 2) if sentiments else 0
                        total_reviews = len(sentiments)

                        # Sentiment percentages
                        positive_pct = round(positive_count / total_reviews * 100, 1) if total_reviews else 0
                        neutral_pct = round(neutral_count / total_reviews * 100, 1) if total_reviews else 0
                        negative_pct = round(negative_count / total_reviews * 100, 1) if total_reviews else 0

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
                            "Reviews": r["Reviews"],
                            "Positive %": r["Positive %"],
                            "Neutral %": r["Neutral %"],
                            "Negative %": r["Negative %"]
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
                    st.markdown(f"""
                        <div style="background-color: {color}; border-radius: 15px; padding: 20px; text-align: center; box-shadow: 0 4px 8px rgba(0,0,0,0.2); color: black; font-weight: bold;">
                            <div style="font-size: 22px; margin-bottom: 10px;">{medal}</div>
                            <div style="font-size: 18px; margin-bottom: 8px;">{r['Restaurant']}</div>
                            <div style="font-size: 15px; margin-bottom: 8px;">{r['Address']}</div>
                            <div style="font-size: 16px;">{r['Stars']} ({r['Rating']})</div>
                            <div style="font-size: 14px; margin-top: 10px; text-align: left;">
                                Positive: {r['Positive %']}%<br>
                                Neutral: {r['Neutral %']}%<br>
                                Negative: {r['Negative %']}%
                            </div>
                        </div>
                    """, unsafe_allow_html=True)

        # Plot sentiment breakdown bar chart for top 3 restaurants
        st.divider()
        st.subheader("📊 Sentiment Breakdown of Top 3 Restaurants")

        fig, ax = plt.subplots()
        labels = [r["Restaurant"] for r in top3]
        positives = [r["Positive %"] for r in top3]
        neutrals = [r["Neutral %"] for r in top3]
        negatives = [r["Negative %"] for r in top3]

        width = 0.2
        x = range(len(top3))

        ax.bar(x, positives, width, label="Positive %", color="green")
        ax.bar([p + width for p in x], neutrals, width, label="Neutral %", color="gray")
        ax.bar([p + width*2 for p in x], negatives, width, label="Negative %", color="red")

        ax.set_xticks([p + width for p in x])
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 100)
        ax.set_ylabel("Percentage")
        ax.legend()
        ax.set_title("Sentiment Breakdown for Top 3 Restaurants")

        st.pyplot(fig)

        st.divider()
        top = max(st.session_state.results, key=lambda x: x["Rating"])
        st.metric(label="🏆 Top Pick", value=top["Restaurant"], delta=f"{top['Rating']} ⭐")

        top_pick = {
            "Restaurant": top["Restaurant"],
            "Rating": top["Rating"],
            "Address": top["Address"],
            "Food": food,
            "Location": location
        }
        append_history(top_pick)

        st.divider()
        st.subheader("📸 Restaurant Highlights")

        cols = st.columns(2)
        for idx, r in enumerate(sorted(st.session_state.results, key=lambda x: x["Rating"], reverse=True)):
            with cols[idx % 2]:
                st.markdown(f"### {r['Restaurant']}")
                st.markdown(f"**📍 Address:** {r['Address']}")
                st.markdown(f"**⭐ Rating:** {r['Rating']} ({r['Reviews']} reviews)")
                if r["Image"]:
                    st.markdown(f"""
                        <div style="width: 100%; height: 220px; overflow: hidden; border-radius: 10px; margin-bottom: 10px;">
                            <img src="{r['Image']}" style="width: 100%; height: 100%; object-fit: cover;" />
                        </div>
                    """, unsafe_allow_html=True)
                if r["Tips"]:
                    st.markdown("💬 **Reviews:**")
                    for tip in r["Tips"]:
                        st.markdown(f"• _{tip}_")
                st.markdown("---")

# -------- PAGE: Deep Learning --------
elif st.session_state.page == "Deep Learning":
    st.title("🤖 AI Sentiment Classifier")
    st.markdown("Try the sentiment classifier on your own text.")

    classifier = get_classifier()
    input_text = st.text_area("Enter review text", height=150)

    if st.button("Analyze Sentiment"):
        if input_text.strip():
            with st.spinner("Analyzing..."):
                result = classifier(input_text[:512])[0]
                label = result["label"]
                score = round(result["score"], 3)
                st.success(f"Sentiment: {label} (Confidence: {score})")
        else:
            st.warning("Please enter some text for analysis.")

# -------- PAGE: History --------
elif st.session_state.page == "History":
    st.title("🕘 Search History")
    history_data = read_history()
    if history_data:
        df_hist = pd.DataFrame(history_data)
        df_hist.index += 1
        st.dataframe(df_hist, use_container_width=True)
    else:
        st.info("No history found.")

# -------- PAGE: About --------
elif st.session_state.page == "About":
    st.title("ℹ️ About This App")
    st.markdown("""
    This app recommends restaurants based on Foursquare's Places API combined with AI-powered sentiment analysis
    of real user reviews (tips). It is built with Streamlit and HuggingFace Transformers.
    
    Developed by Your Name.
    """)

