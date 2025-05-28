import streamlit as st
import requests
import pandas as pd
from transformers import pipeline

# Page setup
st.set_page_config(page_title="🍽️ Restaurant Recommender", layout="wide")
st.title("🍽️ AI Restaurant Recommender")
st.markdown("Find top-rated restaurants near you using **Foursquare** and **AI sentiment analysis** of real user reviews.")

# Session state
if "results" not in st.session_state:
    st.session_state.results = None
    st.session_state.df = None

# Input section
with st.container():
    col1, _ = st.columns([2, 5])  # col1 = input, rest = empty space
    with col1:
        food = st.text_input("🍕 Food Type", placeholder="e.g., Sushi, Jollof, Pizza")

with st.container():
    col1, _ = st.columns([2, 5])
    with col1:
        location = st.text_input("📍 Location", placeholder="e.g., Lagos, Nigeria")

# API Key
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

                # Fetch tips
                tips_url = f"https://api.foursquare.com/v3/places/{fsq_id}/tips"
                tips_res = requests.get(tips_url, headers=headers)
                tips = tips_res.json()
                review_texts = [tip["text"] for tip in tips[:5]] if tips else []

                sentiments = []
                for tip in review_texts:
                    result = classifier(tip[:512])[0]
                    stars = int(result["label"].split()[0])
                    sentiments.append(stars)

                # Fetch image
                photo_url = ""
                photo_api = f"https://api.foursquare.com/v3/places/{fsq_id}/photos"
                photo_res = requests.get(photo_api, headers=headers)
                photos = photo_res.json()
                if photos:
                    photo = photos[0]
                    photo_url = f"{photo['prefix']}original{photo['suffix']}"

                if sentiments:
                    avg_rating = round(sum(sentiments) / len(sentiments), 2)
                    results.append({
                        "Restaurant": name,
                        "Address": address,
                        "Rating": avg_rating,
                        "Stars": "⭐" * int(round(avg_rating)),
                        "Reviews": len(sentiments),
                        "Image": photo_url,
                        "Tips": review_texts if review_texts else []
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

                df.index = df.index + 1  # Make index start from 1
                st.session_state.results = results
                st.session_state.df = df
            else:
                st.warning("Found restaurants but no reviews available.")

# Display results
if st.session_state.results:
    st.divider()
    st.subheader("📊 Restaurant Table")
    st.dataframe(st.session_state.df, use_container_width=True)

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
                st.markdown(
                    f"""
                    <div style="width: 100%; height: 220px; overflow: hidden; border-radius: 10px; margin-bottom: 10px;">
                        <img src="{r['Image']}" style="width: 100%; height: 100%; object-fit: cover;" />
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            tips = r.get("Tips", [])[:2]  # show first 2 reviews
            if tips:
                st.markdown("💬 **Reviews:**")
                for tip in tips:
                    st.markdown(f"• _{tip}_")

            st.markdown("---")
