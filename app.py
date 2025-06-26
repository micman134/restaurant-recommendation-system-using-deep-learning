import streamlit as st
import requests
import pandas as pd
from transformers import pipeline

# Sentiment classifier cached for speed
@st.cache_resource(show_spinner=False)
def get_classifier():
    return pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

# Foursquare API search
def search_foursquare(food, location, api_key, limit=10):
    headers = {"accept": "application/json", "Authorization": api_key}
    params = {"query": food, "near": location, "limit": limit}
    res = requests.get("https://api.foursquare.com/v3/places/search", headers=headers, params=params)
    if res.status_code == 200:
        return res.json().get("results", [])
    return []

# Foursquare tips fetch
def get_foursquare_tips(fsq_id, api_key):
    url = f"https://api.foursquare.com/v3/places/{fsq_id}/tips"
    headers = {"accept": "application/json", "Authorization": api_key}
    res = requests.get(url, headers=headers)
    if res.status_code == 200:
        return [t["text"] for t in res.json()[:5]]
    return []

# Yelp business search via RapidAPI
def search_yelp_business(food, location, limit=10):
    url = "https://yelp-business-reviews.p.rapidapi.com/search"
    headers = {
        "x-rapidapi-key": st.secrets["RAPIDAPI_YELP"]["key"],
        "x-rapidapi-host": st.secrets["RAPIDAPI_YELP"]["host"]
    }
    params = {"location": location, "query": food}
    res = requests.get(url, headers=headers, params=params)
    if res.status_code == 200:
        return res.json().get("businesses", [])[:limit]
    else:
        st.warning(f"Yelp search failed: {res.status_code}")
    return []

# Yelp reviews fetch by business ID via RapidAPI
def get_yelp_reviews_by_id(business_id):
    url = f"https://yelp-business-reviews.p.rapidapi.com/reviews/{business_id}"
    headers = {
        "x-rapidapi-key": st.secrets["RAPIDAPI_YELP"]["key"],
        "x-rapidapi-host": st.secrets["RAPIDAPI_YELP"]["host"]
    }
    res = requests.get(url, headers=headers)
    if res.status_code == 200:
        return [r["text"] for r in res.json().get("reviews", []) if "text" in r]
    else:
        st.warning(f"Failed to get Yelp reviews for business id {business_id}: {res.status_code}")
        return []

# Streamlit UI setup
st.set_page_config(page_title="Restaurant Recommender", layout="wide")
st.title("🍽️ AI Restaurant Recommender (Foursquare + Yelp)")

food = st.text_input("🍕 Food Type", "Pizza")
location = st.text_input("📍 Location", "Lagos, Nigeria")

if st.button("🔍 Search"):
    if not food or not location:
        st.warning("Please enter both food and location.")
    else:
        classifier = get_classifier()
        fsq_api_key = st.secrets["FOURSQUARE_API_KEY"]

        # Fetch restaurants from both sources
        fsq_restaurants = search_foursquare(food, location, fsq_api_key, limit=10)
        yelp_restaurants = search_yelp_business(food, location, limit=10)

        combined_results = []

        # Process Foursquare restaurants
        for r in fsq_restaurants:
            name = r.get("name", "Unknown")
            fsq_id = r.get("fsq_id")
            address = r.get("location", {}).get("formatted_address", "Unknown")

            fsq_reviews = get_foursquare_tips(fsq_id, fsq_api_key)
            # No Yelp reviews to avoid duplicates
            all_reviews = fsq_reviews

            sentiments = []
            for review in all_reviews[:5]:
                try:
                    result = classifier(review[:512])[0]
                    stars = int(result["label"].split()[0])
                    sentiments.append(stars)
                except Exception:
                    continue

            avg_rating = round(sum(sentiments) / len(sentiments), 2) if sentiments else 0

            combined_results.append({
                "Restaurant": name,
                "Address": address,
                "Rating": avg_rating,
                "Reviews": len(all_reviews),
                "Source": "Foursquare",
                "Top Reviews": all_reviews[:2] if all_reviews else ["No reviews"]
            })

        # Process Yelp restaurants
        for r in yelp_restaurants:
            name = r.get("name", "Unknown")
            address = ", ".join(r.get("location", {}).get("display_address", []))
            biz_id = r.get("id")

            yelp_reviews = get_yelp_reviews_by_id(biz_id)
            all_reviews = yelp_reviews

            sentiments = []
            for review in all_reviews[:5]:
                try:
                    result = classifier(review[:512])[0]
                    stars = int(result["label"].split()[0])
                    sentiments.append(stars)
                except Exception:
                    continue

            avg_rating = round(sum(sentiments) / len(sentiments), 2) if sentiments else 0

            combined_results.append({
                "Restaurant": name,
                "Address": address,
                "Rating": avg_rating,
                "Reviews": len(all_reviews),
                "Source": "Yelp",
                "Top Reviews": all_reviews[:2] if all_reviews else ["No reviews"]
            })

        # Sort combined list by rating descending and limit to 20
        combined_results = sorted(combined_results, key=lambda x: x["Rating"], reverse=True)[:20]

        if combined_results:
            df = pd.DataFrame([{
                "Restaurant": r["Restaurant"],
                "Address": r["Address"],
                "Average Rating": r["Rating"],
                "Reviews": r["Reviews"],
                "Source": r["Source"]
            } for r in combined_results])
            df.index += 1
            st.subheader("📊 Combined Results (Foursquare + Yelp)")
            st.dataframe(df, use_container_width=True)

            st.subheader("📝 Sample Reviews")
            for r in combined_results:
                st.markdown(f"### {r['Restaurant']} ({r['Rating']}⭐) — Source: {r['Source']}")
                st.markdown(f"📍 {r['Address']}")
                for review in r["Top Reviews"]:
                    st.markdown(f"- _{review}_")
                st.markdown("---")
        else:
            st.error("No results found. Try different search terms.")
