import streamlit as st
import requests
import pandas as pd
from transformers import pipeline

# UI Setup
st.set_page_config(page_title="🍽️ Restaurant Recommender", layout="wide")
st.title("🍽️ AI Restaurant Recommender")
st.markdown("Discover top-rated restaurants near you based on real reviews using **AI-powered sentiment analysis** and the **Foursquare Places API**.")

# Inputs
food = st.text_input("🍕 What kind of food are you craving?", placeholder="e.g., Jollof, Sushi, Pizza")
location = st.text_input("📍 Where are you located?", placeholder="e.g., Lagos, Nigeria")

# API Key from secrets
api_key = st.secrets.get("FOURSQUARE_API_KEY", "")

if st.button("🔍 Find Restaurants") and food and location and api_key:
    with st.spinner("Finding delicious places..."):

        # Prepare request
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
            st.error("❌ No restaurants found. Try different food or location.")
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

            if results:
                st.success(f"✅ Top restaurants for *{food}* in *{location}*")

                # Table View
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
                st.dataframe(df, use_container_width=True)

                # Top Pick
                top = max(results, key=lambda x: x["Rating"])
                st.metric(label="🏆 Top Pick", value=top["Restaurant"], delta=f"{top['Rating']} ⭐")

                # Image Preview
                st.subheader("📸 Preview with Images")
                for r in sorted(results, key=lambda x: x["Rating"], reverse=True):
                    with st.container():
                        st.markdown(f"#### {r['Restaurant']}")
                        st.write(f"📍 {r['Address']}")
                        st.write(f"{r['Stars']} — {r['Reviews']} reviews")
                        if r["Image"]:
                            st.image(r["Image"], width=400)
                        st.markdown("---")
            else:
                st.warning("Found restaurants but no reviews to analyze.")
