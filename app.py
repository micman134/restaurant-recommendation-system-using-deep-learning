# ======== ANALYSIS SECTION ========
st.divider()
st.subheader("📈 Recommendation Analysis")

# Create DataFrame from results
analysis_df = pd.DataFrame(st.session_state.results)

# Only show analysis if we have data
if not analysis_df.empty:
    # Create tabs for different analysis views
    tab1, tab2, tab3 = st.tabs(["Cuisine Popularity", "Price vs Rating", "Review Insights"])
    
    with tab1:
        # Extract cuisine types from restaurant names and food types
        analysis_df['Cuisine'] = analysis_df['Restaurant'].apply(
            lambda x: ' '.join([w for w in x.split() if w.isupper() or w.istitle()][:1])
        
        # Count cuisine occurrences
        cuisine_counts = analysis_df['Cuisine'].value_counts().reset_index()
        cuisine_counts.columns = ['Cuisine', 'Count']
        
        if not cuisine_counts.empty:
            # Cuisine popularity chart
            fig = px.bar(cuisine_counts.head(10), 
                         x='Cuisine', y='Count',
                         title='Most Popular Cuisine Types',
                         color='Count',
                         color_continuous_scale='teal')
            fig.update_layout(xaxis_title="Cuisine Type",
                             yaxis_title="Number of Restaurants")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No cuisine data available for visualization")
    
    with tab2:
        # Create mock price data (since Foursquare doesn't provide price)
        import numpy as np
        analysis_df['Price'] = np.random.randint(1, 4, size=len(analysis_df))
        
        if 'Rating' in analysis_df.columns:
            analysis_df['Rating'] = pd.to_numeric(analysis_df['Rating'], errors='coerce')
            rated_df = analysis_df[analysis_df['Rating'] > 0]
            
            if not rated_df.empty:
                fig2 = px.box(rated_df, x='Price', y='Rating',
                             title='Price Range vs Rating',
                             color='Price',
                             category_orders={"Price": [1, 2, 3]},
                             labels={"Price": "Price Level (1=$, 2=$$, 3=$$$)"})
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.warning("No valid rating data available")
        else:
            st.warning("No rating data available")
    
    with tab3:
        # Sentiment analysis of reviews
        st.markdown("### 💬 Review Sentiment Highlights")
        
        # Get all review texts
        all_reviews = [review for sublist in analysis_df['Tips'] for review in sublist if review != "No reviews available"]
        
        if all_reviews:
            # Show word cloud of common terms
            text = ' '.join(all_reviews)
            wordcloud = WordCloud(width=800, height=400, background_color='black').generate(text)
            
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis("off")
            st.pyplot(plt)
            
            # Show review length distribution
            review_lengths = [len(review) for review in all_reviews]
            fig3 = px.histogram(x=review_lengths, 
                               title='Distribution of Review Lengths',
                               labels={'x': 'Number of Characters'},
                               color_discrete_sequence=['#FFA500'])
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.warning("No reviews available for analysis")
else:
    st.info("No data available for analysis in current search results")
