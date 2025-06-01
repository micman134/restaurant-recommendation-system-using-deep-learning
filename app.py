import streamlit as st
import requests
import pandas as pd
from transformers import pipeline

# Set page config at the top
st.set_page_config(page_title="🍽️ Restaurant Recommender", layout="wide")

# Custom CSS for sticky header, responsive nav, and hidden Streamlit icons
st.markdown("""
    <style>
    /* Hide default Streamlit menu and icons */
    #MainMenu, footer, header {visibility: hidden;}
    .stDeployButton, .st-emotion-cache-13ln4jf, button[kind="icon"] {
        display: none !important;
    }

    /* Sticky dark header */
    .custom-header {
        position: sticky;
        top: 0;
        z-index: 999;
        background-color: #111;
        color: white;
        padding: 1rem 2rem;
        font-size: 18px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
    }

    .custom-header .brand {
        display: flex;
        align-items: center;
        font-size: 20px;
        font-weight: bold;
    }

    .custom-header .brand img {
        height: 32px;
        width: 32px;
        margin-right: 10px;
        border-radius: 50%;
    }

    .custom-header .nav-links {
        display: flex;
        flex-wrap: wrap;
        margin-top: 10px;
    }

    .custom-header .nav-links a {
        color: white;
        text-decoration: none;
        margin-left: 2rem;
        font-weight: bold;
    }

    .custom-header .nav-links a:hover {
        text-decoration: underline;
    }

    /* Mobile nav */
    @media screen and (max-width: 768px) {
        .custom-header {
            flex-direction: column;
            align-items: flex-start;
        }
        .custom-header .nav-links {
            margin-top: 10px;
            flex-direction: column;
        }
        .custom-header .nav-links a {
            margin: 0.5rem 0;
        }
    }

    .custom-footer {
        text-align: center;
        font-size: 14px;
        margin-top: 50px;
        padding: 20px;
        color: #aaa;
    }
    </style>

    <div class="custom-header">
        <div class="brand">
            <img src="https://cdn-icons-png.flaticon.com/512/3075/3075977.png" />
            🍽️ AI Restaurant Recommender
        </div>
        <div class="nav-links">
            <a href="#">Recommend</a>
            <a href="#">Deep Learning</a>
            <a href="#">About</a>
        </div>
    </div>
""", unsafe_allow_html=True)

# App description
st.markdown("Find top-rated restaurants near you using **Foursquare** and **AI sentiment analysis** of real user reviews.")

# Session state init
if "results" not in st.session_state:
    st.session_state.results = None
    st.session_state.df = None

# Input fields
with st.container():
    col1, _ = st.columns([1, 1])
    with col1:
        food = st.text_input("🍕 Food Type", placeholder="e.g., Sushi, Jollof, Pizza")

with st.container():
    col1, _ = st.columns([1, 1])
    with col1:
        location = st.text_input("📍 Location", placeholder="e.g., Lagos, Nigeria")

# Foursquare API Key
api_key = st.secrets.get("FOURSQUARE_API_KEY", "")

# Search logic
if st.button("🔍 Search") and food and location and api_key:
    st.session_state.results = None
    st.session_state.df = None

    with st.spinner("Searching and analyzing reviews..."):
        headers = {
            "accept": "application/json",
            "Authorization": api_key
        }
        params = {
            "query": food,
            "near": location,
            "limit": 10
        }

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
                    "Rating": avg_rating,
                    "Stars": "⭐" * int(round(avg_rating)),
                    "Reviews": len(sentiments),
                    "Image": photo_url,
                    "Tips": review_texts[:2]
                })

            if results:
                df = pd.DataFrame([
                    {
                        "Restaurant": r["Restaurant"],
                        "Address": r["Address"],
                        "Average Rating": r["Rating"],
                        "Stars": r["Stars"],
                        "Reviews": r["Reviews"]
                    }
                    for r in results
                ])
                df.index = df.index + 1
                st.session_state.results = results
                st.session_state.df = df
            else:
                st.warning("Found restaurants, but no reviews available.")

# Display results
if st.session_state.results:
    st.divider()
    st.subheader("📊 Restaurant Table")
    st.dataframe(st.session_state.df, use_container_width=True)

    st.divider()
    st.subheader("🏅 Top 3 Picks")

    top3 = sorted(st.session_state.results, key=lambda x: x["Rating"], reverse=True)[:3]
    cols = st.columns(3)
    medals = ["🥇 1st", "🥈 2nd", "🥉 3rd"]
    colors = ["#FFD700", "#C0C0C0", "#CD7F32"]

    for i, (col, medal, color) in enumerate(zip(cols, medals, colors)):
        if i < len(top3):
            r = top3[i]
            with col:
                st.markdown(f"""
                <div style="
                    background-color: {color};
                    border-radius: 15px;
                    padding: 20px;
                    text-align: center;
                    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
                    color: black;
                    font-weight: bold;
                ">
                    <div style="font-size: 22px; margin-bottom: 10px;">{medal}</div>
                    <div style="font-size: 18px; margin-bottom: 8px;">{r['Restaurant']}</div>
                    <div style="font-size: 16px;">{r['Stars']} ({r['Rating']})</div>
                </div>
                """, unsafe_allow_html=True)

    top = max(st.session_state.results, key=lambda x: x["Rating"])
    st.metric(label="🏆 Top Pick", value=top["Restaurant"], delta=f"{top['Rating']} ⭐")

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

# Footer
st.markdown("""
    <div class="custom-footer">
        Built with ❤️ using Streamlit, Foursquare, and HuggingFace
    </div>
""", unsafe_allow_html=True)
