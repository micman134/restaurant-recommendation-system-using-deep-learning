import streamlit as st
import requests
import pandas as pd
from transformers import pipeline

# Load sentiment analysis model (cached for performance)
@st.cache_resource(show_spinner=False)
def get_classifier():
    return pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

# Fetch Yelp reviews via RapidAPI (2-step fallback)
def search_yelp_business(food, location):
    url = "https://yelp-business-reviews.p.rapidapi.com/search"
    headers = {
        "x-rapidapi-key": st.secrets["RAPIDAPI_YELP"]["key"],
        "x-rapidapi-host": st.secrets["RAPIDAPI_YELP"]["host"]
    }
    params = {"location": location, "query": food}
    res = requests.get(url, headers=headers, params=params)
    return res.json().get("businesses", [])

def get_yelp_reviews_by_id(business_id):
    url = f"https://yelp-business-reviews.p.rapidapi.com/reviews/{business_id}"
    headers = {
        "x-rapidapi-key": st.secrets["RAPIDAPI_YELP"]["key"],
        "x-rapidapi-host": st.secrets["RAPIDAPI_YELP"]["host"]
    }
    res = requests.get(url, headers=headers)
    return [r["text"] for r in res.json().get("reviews", []) if "text" in r]

def get_yelp_reviews(food, location):
    reviews = []
    businesses = search_yelp_business(food, location)
    for biz in businesses[:1]:  # only take the first business for speed
        reviews += [r["text"] for r in biz.get("reviews", [])]
        if not reviews:
            reviews += get_yelp_reviews_by_id(biz["id"])
    return reviews

# Foursquare search and tip fetch
def search_foursquare(food, location, api_key):
    headers = {"accept": "application/json", "Authorization": api_key}
    params = {"query": food, "near": location, "limit": 10}
    res = requests.get("https://api.foursquare.com/v3/places/search", headers=headers, params=params)
    return res.json().get("results", [])

def get_foursquare_tips(fsq_id, api_key):
    url = f"https://api.foursquare.com/v3/places/{fsq_id}/tips"
    headers = {"accept": "application/json", "Authorization": api_key}
    res = requests.get(url, headers=headers)
    return [t["text"] for t in res.json()[:5]] if res.ok else []

# Streamlit UI
st.set_page_config(page_title="Restaurant Recommender", layout="wide")
st.title("🍽️ AI Restaurant Recommender")
st.markdown("Uses **Foursquare + Yelp** and **BERT sentiment analysis** for smart restaurant picks.")

food = st.text_input("🍕 Food Type", "Pizza")
location = st.text_input("📍 Location", "Lagos, Nigeria")

if st.button("🔍 Search"):
    if not food or not location:
        st.warning("Please enter both food and location.")
    else:
        classifier = get_classifier()
        api_key = st.secrets["FOURSQUARE_API_KEY"]
        restaurants = search_foursquare(food, location, api_key)
        results = []

        for r in restaurants:
            name = r["name"]
            fsq_id = r["fsq_id"]
            address = r.get("location", {}).get("formatted_address", "Unknown")

            # Reviews from Foursquare
            fsq_reviews = get_foursquare_tips(fsq_id, api_key)
            # Reviews from Yelp
            yelp_reviews = get_yelp_reviews(name, location)
            all_reviews = fsq_reviews + yelp_reviews

            sentiments = []
            for review in all_reviews[:5]:  # limit to 5 for performance
                try:
                    result = classifier(review[:512])[0]
                    stars = int(result["label"].split()[0])
                    sentiments.append(stars)
                except Exception:
                    continue

            avg_rating = round(sum(sentiments) / len(sentiments), 2) if sentiments else 0

            results.append({
                "Restaurant": name,
                "Address": address,
                "Rating": avg_rating,
                "Reviews": len(all_reviews),
                "Top Reviews": all_reviews[:2] if all_reviews else ["No reviews"]
            })

        if results:
            df = pd.DataFrame([{
                "Restaurant": r["Restaurant"],
                "Address": r["Address"],
                "Average Rating": r["Rating"],
                "Reviews": r["Reviews"]
            } for r in results])
            df.index += 1
            st.subheader("📊 Results")
            st.dataframe(df, use_container_width=True)

            st.subheader("📝 Reviews")
            for r in results:
                st.markdown(f"### {r['Restaurant']} ({r['Rating']}⭐)")
                st.markdown(f"📍 {r['Address']}")
                for review in r["Top Reviews"]:
                    st.markdown(f"- _{review}_")
                st.markdown("---")
        else:
            st.error("No results found. Try different search terms.")
