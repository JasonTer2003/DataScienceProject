import streamlit as st

# Config
st.set_page_config(layout='wide')
image = "transparent_logo.png"
st.logo(image, icon_image=image)
st.sidebar.markdown("Welcome to Mumo!")
st.sidebar.markdown("A Music Recommendation System Based on Mood Prediction")

# Paging
welcome_page = st.Page('pages/welcome.py', title='Welcome', icon=':material/menu_book:')
mood_based_page = st.Page('pages/mood_based.py', title='Mood-Based Recommender', icon=':material/mood:')
#'\U0001F3B5'
similar_songs_page = st.Page('pages/similar_songs.py', title='Similar Songs Recommender', icon=':material/music_note:')
visualization_page = st.Page('pages/visualization.py', title='Visualization', icon=':material/thumb_up:')

pg = st.navigation([welcome_page, mood_based_page, similar_songs_page, visualization_page])
pg.run()

# st.title(':material/monitoring: MuMo')
# st.divider()
