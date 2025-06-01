import streamlit as st
import requests
import pandas as pd
from transformers import pipeline

# Set page configuration — must be FIRST
st.set_page_config(page_title="🍽️ Restaurant Recommender", layout="wide")

# Inject custom CSS to hide default Streamlit UI and style sidebar/footer
st.markdown("""
    <style>
    /* Hide Streamlit default UI */
    #MainMenu, footer, header {visibility: hidden;}
    .stDeployButton, .st-emotion-cache-13ln4jf, button[kind="icon"] {
        display: none !important;
    }

    /* Sidebar styling */
    .css-1d391kg {  /* class for the sidebar */
        background-color: #111 !important;
        color: white !important;
    }
    .css-1d391kg a {
        color: white !important;
        text-decoration: none;
        font-weight: bold;
        display: block;
        margin: 1rem 0;
        font-size: 18px;
    }
    .css-1d391kg a:hover {
        text-decoration: underline;
    }

    /* Custom footer */
    .custom-footer {
        text-align: center;
        font-size: 14px;
        margin-top: 50px;
        padding: 20px;
        color: #aaa;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar navigation menu
menu = st.sidebar.radio(
    "Navigation",
    options=["Recommend", "Deep Learning", "About"],
    index=0,
)

# --- Recommend Page ---
if menu == "Recommend":
    st.title("🍽️ AI Restaurant Recommender")
    st.markdown("Find top-rated restaurants near you using **Foursquare** and **AI sentiment analysis** of real user reviews.")

    # Session state
    if "results" not in st.session_state:
        st.session_state.results = None
        st.session_state.df = None

    # Inputs
    with st.container():
        col1, _ = st.columns([1, 1])
        with col1:
            food = st.text_input("🍕 Food Type", placeholder="e.g., Sushi, Jollof, Pizza")

    with st.container():
        col1, _ = st.columns([1, 1])
        with col1:
            location = st.text_input("📍 Location", placeholder="e.g., Lagos, Nigeria")

    # API key
    api_key = st.secrets.get("FOURSQUARE_API_KEY", "")

    # Search action
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

                # Save results
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

        # Top 3 picks
        top3 = sorted(st.session_state.results, key=lambda x: x["Rating"], reverse=True)[:3]

        st.divider()
        st.subheader("🏅 Top 3 Picks")

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
                            <div style="font-size: 16px;">{r['Stars']} ({r['Rating']})</div>
                        </div>
                    """, unsafe_allow_html=True)

        top = max(st.session_state.results, key=lambda x: x["Rating"])
        st.metric(label="🏆 Top Pick", value=top["Restaurant"], delta=f"{top['Rating']} ⭐")

        # Images and reviews
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

# --- Deep Learning Page ---
elif menu == "Deep Learning":
    st.title("🤖 Deep Learning in Restaurant Recommendation")
    st.markdown("""
    ### How AI Sentiment Analysis Works

    Our app leverages state-of-the-art deep learning models to analyze user reviews and predict sentiment ratings.

    - We use a **BERT-based multilingual sentiment classifier** (`nlptown/bert-base-multilingual-uncased-sentiment`) to understand the emotional tone of reviews.
    - BERT (Bidirectional Encoder Representations from Transformers) is a transformer-based model pre-trained on large corpora of text data.
    - It reads each review and predicts a star rating from 1 to 5 based on the sentiment.
    - By averaging ratings from multiple reviews, the app generates an accurate sentiment-based score for each restaurant.

    This approach helps us provide recommendations beyond just numeric ratings by interpreting the actual opinions of users.

    ### Benefits of Deep Learning in Our App:
    - Understands nuanced language in multiple languages.
    - Adapts to varying review lengths and styles.
    - Provides more human-like understanding than simple keyword matching.

    ### Learn more:
    - [BERT Paper (2018)](https://arxiv.org/abs/1810.04805)
    - [nlptown/bert-base-multilingual-uncased-sentiment model](https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment)
    """)

# --- About Page ---
else:
    st.title("ℹ️ About AI Restaurant Recommender")
    st.markdown("""
    **AI Restaurant Recommender** is a demo app built to showcase how AI and location data can combine to provide personalized restaurant suggestions.

    - Developed using **Streamlit** for an easy-to-use web app interface.
    - Uses the **Foursquare Places API** to get restaurant information and reviews.
    - Employs **transformers** deep learning models for sentiment analysis.
    - Designed and built by a passionate developer who loves food and AI!

    #### Contact
    - Email: your.email@example.com
    - GitHub: [github.com/yourprofile](https://github.com/yourprofile)
    - LinkedIn: [linkedin.com/in/yourprofile](https://linkedin.com/in/yourprofile)

    Thanks for trying out the app! 🍽️
    """)

# Custom footer
st.markdown("""
<footer class="custom-footer">
    &copy; 2025 AI Restaurant Recommender Final Project using Fourquare API and Deep Learning
</footer>
""", unsafe_allow_html=True)
