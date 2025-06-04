# Save as app.py and run using: streamlit run app.py

import streamlit as st
import requests
import pandas as pd
from transformers import pipeline
import matplotlib.pyplot as plt

# UI setup
st.set_page_config(page_title="🍽️ AI Restaurant Recommender", layout="wide")
st.markdown("<style>#MainMenu, footer, header {visibility: hidden;}</style>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("🍴 Menu")
    page = st.radio("Navigate", ["Recommend", "Deep Learning", "About"])

# Recommender Page
if page == "Recommend":
    st.title("🍽️ AI Restaurant Recommender")
    food = st.text_input("🍕 Food Type", placeholder="e.g. Sushi, Jollof, Pizza")
    location = st.text_input("📍 Location", placeholder="e.g. Lagos, Nigeria")
    diet_tags = st.multiselect("🥗 Dietary Preferences", ["Vegan", "Vegetarian", "Halal", "Kosher", "Gluten-Free", "Dairy-Free", "Pescatarian"])

    api_key = st.secrets.get("FOURSQUARE_API_KEY", "")  # Replace with your API key or use st.text_input for demo

    if st.button("🔍 Search") and food and location and api_key:
        with st.spinner("Searching and analyzing..."):
            headers = {"accept": "application/json", "Authorization": api_key}
            params = {"query": food, "near": location, "limit": 10}
            res = requests.get("https://api.foursquare.com/v3/places/search", headers=headers, params=params)
            restaurants = res.json().get("results", [])
            
            classifier = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
            results = []

            for r in restaurants:
                fsq_id = r["fsq_id"]
                name = r["name"]
                address = r["location"].get("formatted_address", "Unknown")

                tips_url = f"https://api.foursquare.com/v3/places/{fsq_id}/tips"
                tips_res = requests.get(tips_url, headers=headers)
                tips = tips_res.json()
                texts = [t["text"] for t in tips[:5]] if tips else []

                sentiments = [int(classifier(text[:512])[0]["label"].split()[0]) for text in texts] if texts else []
                avg_rating = round(sum(sentiments) / len(sentiments), 2) if sentiments else 0

                photo_url = ""
                photo_res = requests.get(f"https://api.foursquare.com/v3/places/{fsq_id}/photos", headers=headers)
                photos = photo_res.json()
                if photos:
                    photo = photos[0]
                    photo_url = f"{photo['prefix']}original{photo['suffix']}"

                results.append({
                    "Restaurant": name,
                    "Address": address,
                    "Rating": avg_rating,
                    "Stars": "⭐" * int(round(avg_rating)),
                    "Reviews": len(sentiments),
                    "Image": photo_url,
                    "Tips": texts
                })

            # Filter by diet tags
            def diet_match(r, tags):
                combined = " ".join(r["Tips"]).lower()
                return any(tag.lower() in combined for tag in tags)

            filtered = [r for r in results if diet_match(r, diet_tags)] if diet_tags else results

            df = pd.DataFrame([{
                "Restaurant": r["Restaurant"],
                "Address": r["Address"],
                "Rating": r["Rating"],
                "Stars": r["Stars"],
                "Reviews": r["Reviews"]
            } for r in filtered])
            df.index += 1

            st.success(f"Found {len(filtered)} result(s).")
            st.dataframe(df, use_container_width=True)

            # Show Top 3
            st.subheader("🏅 Top 3 Recommendations")
            top3 = sorted(filtered, key=lambda x: x["Rating"], reverse=True)[:3]
            for i, r in enumerate(top3, start=1):
                st.markdown(f"**{i}. {r['Restaurant']}** - {r['Stars']} - 📍 {r['Address']}")

            # Images
            st.subheader("📸 Highlights")
            cols = st.columns(2)
            for idx, r in enumerate(filtered):
                with cols[idx % 2]:
                    st.markdown(f"### {r['Restaurant']}")
                    if r["Image"]:
                        st.image(r["Image"], width=400, caption=r["Restaurant"])
                    if r["Tips"]:
                        st.markdown("💬 _" + r["Tips"][0] + "_")
                    st.markdown("---")

# Deep Learning Explanation
elif page == "Deep Learning":
    st.title("🤖 Deep Learning")
    st.write("This app uses BERT sentiment analysis to process Foursquare reviews and rate restaurants.")
    st.write("Model: `nlptown/bert-base-multilingual-uncased-sentiment`")

    st.subheader("📊 Sample Rating Distribution")
    fig, ax = plt.subplots()
    ax.bar(["1⭐", "2⭐", "3⭐", "4⭐", "5⭐"], [15, 30, 50, 80, 100], color="skyblue")
    st.pyplot(fig)

# About Page
else:
    st.title("ℹ️ About This App")
    st.markdown("""
    This restaurant recommender uses:
    - 🧠 Deep Learning for sentiment analysis
    - 📍 Foursquare API for data
    - 🥗 Filters based on your dietary preferences
    """)

    st.code("Built with Python + Streamlit + HuggingFace 🤗")
