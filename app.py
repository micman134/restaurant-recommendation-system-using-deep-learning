import streamlit as st
import requests
import pandas as pd
from transformers import pipeline

# Page setup
st.set_page_config(page_title="🍽️ Restaurant Recommender", layout="wide")
st.title("🍽️ AI Restaurant Recommender")
st.markdown("Discover top-rated restaurants near you based on real reviews using **AI-powered sentiment analysis** and the **Foursquare Places API**.")

# Session state to persist results
if "results" not in st.session_state:
    st.session_state.results = None
    st.session_state.df = None

# Inputs
food = st.text_input("🍕 What kind of food are you craving?", placeholder="e.g., Jollof, Sushi, Pizza")
location = st.text_input("📍 Where are you located?", placeholder="e.g., Lagos, Nigeria")

# API Key from secrets
api_key = st.secrets.get("FOURSQUARE_API_KEY", "")

# Search
if st.button("🔍 Find Restaurants") and food and location and api_key:
    with st.spinner("Finding delicious places..."):

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
            st.error("❌ No restaurants found. Try a different search.")
            st.session_state.results = None
            st.session_state.df = None
        else:
            classifier = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
            results = []

            for r in restaurants:
                fsq_id = r['fsq_id']
                name = r['name']
                address = r['location'].get('formatted_address', 'Unknown')

                # Get tips
                tips_url = f"https://api.foursquare.com/v3/places/{fsq_id}/tips"
                tips_res = requests.get(tips_url, headers=headers)
                tips = tips_res.json()

                sentiments = []
                for tip in tips:
                    result = classifier(tip["text"][:512])[0]
                    stars = int(result["label"].split()[0])
                    sentiments.append(stars)

                # Get image
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
                        "Image": photo_url
                    })

            # Save to session_state
            if results:
                df = pd.DataFrame([
                    {
                        "Restaurant": r["Restaurant"],
                        "Address": r["Address"],
                        "Average Rating": r["Rating"],
                        "Star Visual": r["Stars"],
                        "Reviews": r["Reviews"]
                    }
                    for r in results
                ])
                st.session_state.results = results
                st.session_state.df = df
            else:
                st.warning("Restaurants found, but no reviews to analyze.")
                st.session_state.results = None
                st.session_state.df = None

# Always display results if available
if st.session_state.results:
    st.success("🍽️ Previous Search Results")
    
    st.dataframe(st.session_state.df, use_container_width=True)

    # Highlight Top Pick
    top = max(st.session_state.results, key=lambda x: x["Rating"])
    st.metric(label="🏆 Top Pick", value=top["Restaurant"], delta=f"{top['Rating']} ⭐")

    # Images + details
    st.subheader("📸 Preview with Images")
    for r in sorted(st.session_state.results, key=lambda x: x["Rating"], reverse=True):
        with st.container():
            st.markdown(f"#### {r['Restaurant']}")
            st.write(f"📍 {r['Address']}")
            st.write(f"{
