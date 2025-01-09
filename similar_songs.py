import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Cache data loading
@st.cache_data
def load_data():
    df = pd.read_csv(r'C:\Drive D\UM\Y3S1\WIH3001 DSP\Latest Project\df_for_recommender.csv')
    return df

data = load_data()

# Define feature vector extraction function
def get_feature_vector(song_name):
    """
    Retrieves the feature vector for a given song by its name.
    
    Parameters:
    - song_name (str): Name of the song to search.

    Returns:
    - feature_vector (numpy.ndarray): Feature vector of the song.
    """
    song = data[data['name'].str.lower() == song_name.lower()]
    if song.empty:
        raise ValueError("Sorry, the song is not found in our database. Try searching for songs available in database!")
    
    # Select only numeric columns, excluding non-numeric data like song names, IDs, etc.
    numeric_columns = song.select_dtypes(include=['number']).columns.tolist()

    # Now, select only the numeric columns for the feature vector
    feature_vector = song[numeric_columns].values
    
    # Reshaping to ensure it's 2D with shape (1, n)
    feature_vector = feature_vector.reshape(1, -1)
    return feature_vector

# Similar songs recommendation function
def get_similar_songs(song_name, top_n):
    """
    Finds and recommends songs similar to the given song name.
    
    Parameters:
    - song_name (str): Name of the song to find similarities.
    - top_n (int): Number of similar songs to return.

    Returns:
    - similar_songs (DataFrame): DataFrame containing similar songs and their popularity.
    """
    # Get the feature vector for the input song
    feature_vector = get_feature_vector(song_name)
    
    # Select numeric data from the whole dataset for similarity calculation
    numeric_data = data.select_dtypes(include=['number']).values
    
    # Ensuring numeric_data is a 2D array
    numeric_data = numeric_data.reshape(-1, numeric_data.shape[1])
    
    # Calculate similarity between the input song's feature vector and all other songs
    similarities = cosine_similarity(numeric_data, feature_vector).flatten()
    
    # Get indices of top N+1 similar songs (excluding the input song itself)
    indices = similarities.argsort()[-(top_n + 1):][::-1][1:]
    
    # Select the top N similar songs and include their details
    similar_songs = data.iloc[indices]
    
    # Filter out songs with zero or negative popularity
    similar_songs = similar_songs[similar_songs['popularity'] > 0].sort_values(by='popularity', ascending=False)
    
    # Add YouTube and Spotify links
    similar_songs['YouTube Link'] = similar_songs['name'].apply(lambda x: f"https://www.youtube.com/results?search_query={x.replace(' ', '+')}")
    similar_songs['Spotify Link'] = similar_songs['name'].apply(lambda x: f"https://open.spotify.com/search/{x.replace(' ', '%20')}")
    
    return similar_songs[['name', 'artists', 'popularity', 'YouTube Link', 'Spotify Link']].head(top_n)

st.title("\U0001F3B6 Similar Songs Recommender")
st.markdown("Bored with your current playlist? Wanna try **exploring new songs**? You've come to the right page!")

# Tabs
song_database_tab, similar_song_tab = st.tabs(['Song List in Database', 'Similar Songs Recommender'])

# Song List in Database
with song_database_tab:

    st.write("Available Songs in Database  (**Recent Top 50 Spotify Songs in Malaysia**):")
    st.dataframe(data[['name', 'artists', 'album_name', 'album_release_date', 'popularity']], height=240, use_container_width=True)

    song_name = st.text_input("Not sure if the song you are searching for is here? Type it here:")

    if st.button("Search!"):
        song = data[data['name'].str.lower() == song_name.lower()]
        if song.empty:
            st.write("Sorry, the song is not found in our database. Try searching for songs available in database!")
        else:
            st.write("The song you searched for is in our database!")

# Similar Songs Recommender
with similar_song_tab:
    song_name = st.text_input("Enter a song name:")
    if st.button("Find Similar Songs"):
        try:
            similar = get_similar_songs(song_name, 5)
            st.subheader("🎵 Songs similar to your input:")
            st.dataframe(
                similar,
                column_config={
                    "YouTube Link": st.column_config.LinkColumn("YouTube URL", display_text="YouTube Link"),
                    "Spotify Link": st.column_config.LinkColumn("Spotify URL", display_text="Spotify Link"),
                },
                hide_index=True,
                use_container_width=True
            )
        except Exception as e:
            st.error(str(e))

        # st.write("Are you looking for more similar songs recommendations?")

        # left, right = st.columns(2)

        # if left.button("Yes", icon=":material/check:", use_container_width=True):
        #     left.markdown("You clicked the plain button.")
        # if right.button("No", icon=":material/close:", use_container_width=True):
        #     right.markdown("Thank you. Hope you enjoy the recommended songs!")