import streamlit as st
import requests
import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

# Page config
st.set_page_config(page_title="🍽️ Restaurant Recommender", layout="wide")
st.title("🍽️ AI Restaurant Recommender")
st.markdown("Discover the best-rated restaurants near you using **real reviews**, **AI-powered sentiment analysis**, and the **Foursquare API**.")

# Input form
with st.form("user_input"):
    food = st.text_input("🍕 What kind of food are you craving?", placeholder="e.g., Jollof, Sushi, Pizza")
    location = st.text_input("📍 Where are you located?", placeholder="e.g., Lagos, Nigeria")
    min_rating = st.slider("🌟 Minimum average rating to include", 1.0, 5.0, 3.5, 0.5)
    max_reviews = st.slider("💬 Max number of reviews to show per restaurant", 1, 10, 3)
    submitted = st.form_submit_button("🔍 Find Restaurants")

# API Key from Streamlit secrets
api_key = st.secrets.get("FOURSQUARE_API_KEY", "")

if submitted and food and location and api_key:
    with st.spinner("Fetching restaurant data..."):

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
            translator = pipeline("translation", model="Helsinki-NLP/opus-mt-mul-en")

            data = []
            all_reviews = {}

            for r in restaurants:
                fsq_id = r['fsq_id']
                name = r['name']
                address = r['location'].get('formatted_address', 'Unknown')

                # Fetch tips
                tips_url = f"https://api.foursquare.com/v3/places/{fsq_id}/tips"
                tips_res = requests.get(tips_url, headers=headers)
                tips = tips_res.json()

                sentiments = []
                reviews = []

                for tip in tips:
                    text = tip["text"]
                    result = classifier(text[:512])[0]
                    stars = int(result["label"].split()[0])
                    sentiments.append(stars)

                    # Translate non-English to English
                    try:
                        translated = translator(text[:512])[0]['translation_text']
                    except:
                        translated = text  # fallback to original

                    reviews.append(f"⭐ {stars} – {translated}")

                if sentiments:
                    avg_rating = round(sum(sentiments) / len(sentiments), 2)
                    if avg_rating >= min_rating:
                        data.append({
                            "Restaurant": name,
                            "Address": address,
                            "Average Rating": avg_rating,
                            "Star Visual": "⭐" * int(round(avg_rating)),
                            "Reviews": len(sentiments)
                        })
                        all_reviews[name] = reviews

            if data:
                df = pd.DataFrame(data)
                df_sorted = df.sort_values(by="Average Rating", ascending=False)

                st.success(f"✅ Top {len(df_sorted)} restaurants for *{food}* in *{location}* with ≥ {min_rating}⭐")
                st.dataframe(df_sorted[["Restaurant", "Address", "Average Rating", "Star Visual", "Reviews"]], use_container_width=True)

                # Top Pick
                top = df_sorted.iloc[0]
                st.metric(label="🏆 Top Pick", value=top["Restaurant"], delta=f"{top['Average Rating']} ⭐")

                # CSV download
                csv = df_sorted.to_csv(index=False).encode("utf-8")
                st.download_button("📥 Download CSV", data=csv, file_name="restaurant_recommendations.csv", mime="text/csv")

                # Show reviews
                st.subheader("💬 Sample User Reviews")
                for _, row in df_sorted.iterrows():
                    name = row["Restaurant"]
                    st.markdown(f"### {name}")
                    for quote in all_reviews.get(name, [])[:max_reviews]:
                        st.write(f"• {quote}")
            else:
                st.warning("No restaurants matched your rating filter.")
