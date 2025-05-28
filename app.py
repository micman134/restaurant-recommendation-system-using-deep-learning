if results:
    st.success(f"✅ Top restaurants for *{food}* in *{location}*")

    # Create summary table
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

    # Highlight Top Pick
    top = max(results, key=lambda x: x["Rating"])
    st.metric(label="🏆 Top Pick", value=top["Restaurant"], delta=f"{top['Rating']} ⭐")

    # Show images with details
    st.subheader("📸 Preview with Images")
    for r in sorted(results, key=lambda x: x["Rating"], reverse=True):
        with st.container():
            st.markdown(f"#### {r['Restaurant']}")
            st.write(f"📍 {r['Address']}")
            st.write(f"{r['Stars']} — {r['Reviews']} reviews")
            if r["Image"]:
                st.image(r["Image"], width=400)
            st.markdown("---")
