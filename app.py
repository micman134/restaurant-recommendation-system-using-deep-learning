import streamlit as st
import requests
import pandas as pd
from transformers import pipeline
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(page_title="🍽️ Restaurant Recommender", layout="wide")

# Hide Streamlit UI
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

# Initialize session state
if "page" not in st.session_state:
    st.session_state.page = "Recommend"

# Sidebar
with st.sidebar:
    st.markdown("## 🍽️ Menu")
    if st.button("Recommend"):
        st.session_state.page = "Recommend"
    if st.button("Deep Learning"):
        st.session_state.page = "Deep Learning"
    if st.button("Dietary & History"):
        st.session_state.page = "Dietary"
    if st.button("About"):
        st.session_state.page = "About"

# -------- PAGE: Recommend --------
if st.session_state.page == "Recommend":
    st.title("🍽️ AI Restaurant Recommender")
    st.markdown("Find top-rated restaurants near you using **Foursquare** and **AI sentiment analysis** of real user reviews.")

    if "results" not in st.session_state:
        st.session_state.results = None
        st.session_state.df = None

    with st.container():
        col1, _ = st.columns([1, 1])
        with col1:
            food = st.text_input("🍕 Food Type", placeholder="e.g., Sushi, Jollof, Pizza")
    with st.container():
        col1, _ = st.columns([1, 1])
        with col1:
            location = st.text_input("📍 Location", placeholder="e.g., Lagos, Nigeria")

    api_key = st.secrets.get("FOURSQUARE_API_KEY", "")

    if st.button("🔍 Search") and food and location and api_key:
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
                classifier = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
                results = []

                for r in restaurants:
                    fsq_id = r['fsq_id']
                    name = r['name']
                    address = r['location'].get('formatted_address', 'Unknown')

                    # Fetch reviews
                    tips_url = f"https://api.foursquare.com/v3/places/{fsq_id}/tips"
                    tips_res = requests.get(tips_url, headers=headers)
                    tips = tips_res.json()
                    review_texts = [tip["text"] for tip in tips[:5]] if tips else []

                    sentiments = []
                    for tip in review_texts:
                        result = classifier(tip[:512])[0]
                        stars = int(result["label"].split()[0])
                        sentiments.append(stars)

                    # Fetch photo
                    photo_url = ""
                    photo_api = f"https://api.foursquare.com/v3/places/{fsq_id}/photos"
                    photo_res = requests.get(photo_api, headers=headers)
                    photos = photo_res.json()
                    if photos:
                        photo = photos[0]
                        photo_url = f"{photo['prefix']}original{photo['suffix']}"

                    # Append result
                    avg_rating = round(sum(sentiments) / len(sentiments), 2) if sentiments else 0
                    results.append({
                        "Restaurant": name,
                        "Address": address,
                        "Rating": avg_rating,
                        "Stars": "⭐" * int(round(avg_rating)),
                        "Reviews": len(sentiments),
                        "Image": photo_url,
                        "Tips": review_texts[:2]
                    })

                if results:
                    df = pd.DataFrame([{
                        "Restaurant": r["Restaurant"],
                        "Address": r["Address"],
                        "Average Rating": r["Rating"],
                        "Stars": r["Stars"],
                        "Reviews": r["Reviews"]
                    } for r in results])
                    df.index = df.index + 1
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
                        </div>
                    """, unsafe_allow_html=True)
        st.divider()

        # Top Pick and save to history
        top = max(st.session_state.results, key=lambda x: x["Rating"])
        st.metric(label="🏆 Top Pick", value=top["Restaurant"], delta=f"{top['Rating']} ⭐")

        # Store top pick to history
        if "history" not in st.session_state:
            st.session_state.history = []
        top_pick = {
            "Restaurant": top["Restaurant"],
            "Rating": top["Rating"],
            "Address": top["Address"]
        }
        if top_pick not in st.session_state.history:
            st.session_state.history.append(top_pick)

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
    st.title("🤖 Deep Learning Explained")
    st.markdown("...")


# -------- PAGE: Dietary & History --------
elif st.session_state.page == "Dietary":
    st.title("🥗 History Review")
    st.markdown("Use this section to manage  review your **top restaurant picks** from past recommendations.")

    
   
    st.markdown("---")
    st.subheader("🕒 Review History (Top Picks)")
    if "history" in st.session_state and st.session_state.history:
        hist_df = pd.DataFrame(st.session_state.history)
        hist_df.index += 1
        st.dataframe(hist_df, use_container_width=True)
    else:
        st.info("You haven't made any recommendations yet.")

# -------- PAGE: About --------
elif st.session_state.page == "About":
    st.title("ℹ️ About This App")
    st.markdown("""
    **AI Restaurant Recommender** was built to help you discover great places to eat by combining:
    - Real user reviews,
    - AI sentiment analysis,
    - And Foursquare's extensive location data.

    Thanks for trying out the app! 🍽️
    """)

# Footer
st.markdown("""
    <div class="custom-footer">
        © 2025 AI (Deep Learning) Restaurant Recommender Final Year Project ·
    </div>
""", unsafe_allow_html=True)
