# streamlit_app.py

import streamlit as st
import requests
import pandas as pd
from transformers import pipeline

# UI Setup
st.set_page_config(page_title="🍽️ Restaurant Recommender", layout="centered")
st.title("🍽️ AI-Powered Restaurant Recommender")
st.markdown("Discover top-rated restaurants based on real reviews, powered by **BERT sentiment analysis**.")

# Inputs
food = st.text_input("🍕 What kind of food are you craving?", placeholder="e.g., Jollof, Sushi, Pizza")
location = st.text_input("📍 Where are you located?", placeholder="e.g., Lagos, Nigeria")

# API Key (streamlit cloud secret)
api_key = st.secrets["FOURSQUARE_API_KEY"]

# Run search
if st.button("🔍 Find Restaurants") and food and location:
    with st.spinner("Finding delicious spots..."):

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
            st.error("No restaurants found. Try a different query.")
        else:
            classifier = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
            data = []

            for r in restaurants:
                fsq_id = r['fsq_id']
                name = r['name']
                address = r['location'].get('formatted_address', 'Unknown')
                lat = r['geocodes']['main']['latitude']
                lon = r['geocodes']['main']['longitude']

                tips_url = f"https://api.foursquare.com/v3/places/{fsq_id}/tips"
                tips_res = requests.get(tips_url, headers=headers)
                tips = tips_res.json()

                sentiments = []
                for tip in tips:
                    result = classifier(tip["text"][:512])[0]
                    stars = int(result["label"].split()[0])
                    sentiments.append(stars)

                if sentiments:
                    avg_rating = round(sum(sentiments) / len(sentiments), 2)
                    data.append({
                        "Restaurant": name,
                        "Address": address,
                        "Average Rating": avg_rating,
                        "Star Visual": "⭐" * int(round(avg_rating)),
                        "Reviews": len(sentiments),
                        "lat": lat,
                        "lon": lon
                    })

            if data:
                df = pd.DataFrame(data)
                df_sorted = df.sort_values(by="Average Rating", ascending=False)

                st.success(f"🍽️ Top Restaurants for *{food}* in *{location}*")
                st.dataframe(df_sorted[["Restaurant", "Address", "Average Rating", "Star Visual", "Reviews"]])

                # Map View
                st.subheader("🗺️ Map View")
                st.map(df_sorted.rename(columns={"lat": "latitude", "lon": "longitude"}))

                # Highlight Top Pick
                top = df_sorted.iloc[0]
                st.metric(label="🏆 Top Pick", value=top["Restaurant"], delta=f"{top['Average Rating']} ⭐")
            else:
                st.info("Found restaurants, but no reviews available to analyze.")
