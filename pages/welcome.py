import streamlit as st

st.image(r"C:\Drive D\UM\Y3S1\WIH3001 DSP\Logo\logo.png", width=250)

st.title("Your Music Recommendation System \U0001F3B6")
st.write("""#### Tailored specially for you! Discover music that matches your mood or find similar songs easily. \U0001F3A7""")

st.divider()

st.image(r"C:\Drive D\UM\Y3S1\WIH3001 DSP\Logo\background.jpg", caption="Image generated using DeepAI", use_container_width=True)

st.subheader("Background")
st.write("Mood is highly connected to music where emotional content has become an important factor affecting music selection today. This statement is supported by Bhosale et al. (2024) where most people listen to music that suits their current mood or helps them transition to a different mood.")
st.write("Therefore, this shows how interconnected music and mood are, which leads to the idea of developing a **music recommendation system based on mood prediction**.")
# st.write("In the Mood-Based Recommender Page, describe your feelings in one sentence and we will recommend the top 5 songs that are best suited to your current mood!")
# st.write("Bored with your current listen? Use the Similar Songs Recommender Page to help you find new songs similar to your taste!")
# st.write("Wonder how our system work? Head towards the Visualization Page to find out!")
st.divider()

st.subheader("Problem Statement")
st.write("**1. Existing models have low accuracy**: Bhosale et al. (2024)'s research only achieve an accuracy of 0.76 for their models on test data.")
st.write("\n**2. Designing systems to recommend music that aligns with users' preferences is still limited**.")
st.divider()

st.subheader("Project Objectives")
st.write("**1. To develop a mood prediction model**")
st.write("**2. To evaluate a mood prediction model**")
st.write("**3. To develop a music recommendation system based on mood prediction model**")
st.divider()

st.subheader("Web Application")
st.write("Head towards each page to discover a magical music journey! ✨")
st.divider()

st.subheader("Note!")
st.write("The music recommendation system has carefully included **music therapy**!")
st.write("All songs with explicit lyrics have been removed to promote our listeners' well-being!", icon=':material/verified:')

# Footer
st.divider()
st.write('Copyright :material/copyright: 2024/2025 TER ZHEN HUANG. All Rights Reserved.')