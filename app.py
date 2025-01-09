import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Cache data loading
@st.cache_data
def load_data():
    df = pd.read_csv('df_for_recommender.csv')
    return df

data = load_data()

# Preprocessing function
def preprocessed_data(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    def lemmatization(text):
        text = text.split()
        text = [lemmatizer.lemmatize(word) for word in text]
        return " ".join(text)

    def remove_stop_words(text):
        text = [word for word in text.split() if word not in stop_words]
        return " ".join(text)

    def removing_numbers(text):
        return ''.join([char for char in text if not char.isdigit()])

    def lower_case(text):
        return text.lower()

    def removing_punctuations(text):
        text = re.sub(r'[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def removing_urls(text):
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        return url_pattern.sub('', text)

    text = lower_case(text)
    text = remove_stop_words(text)
    text = removing_numbers(text)
    text = removing_punctuations(text)
    text = removing_urls(text)
    text = lemmatization(text)
    return text

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
        raise ValueError("Song not found in dataset. Only search for songs available in dataset!")
    
    # Select only numeric columns, excluding non-numeric data like song names, IDs, etc.
    numeric_columns = song.select_dtypes(include=['number']).columns.tolist()

    # Now, select only the numeric columns for the feature vector
    feature_vector = song[numeric_columns].values
    
    # Reshaping to ensure it's 2D with shape (1, n)
    feature_vector = feature_vector.reshape(1, -1)
    return feature_vector

# Similar songs recommendation function
def get_similar_songs(song_name, top_n=5):
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
    
    # Return the top N similar songs
    return similar_songs[['name', 'artists', 'popularity', 'YouTube Link', 'Spotify Link']].head(top_n)

import zipfile
import os

# Specify the path to the zip file and the extraction directory
zip_file_path = 'rf_model.zip'
extraction_dir = 'extracted_model/'

# Unzip the file
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extraction_dir)

# Now, load the model from the extracted file
model_path = os.path.join(extraction_dir, 'rf_model.pkl')
with open(model_path, "rb") as file:
    rf_model = pickle.load(file)

with open("tfidf_parameters.pkl", "rb") as file:
    tfidf_vect = joblib.load(file)

# Mood prediction function
def predict_mood(rf_model, text_input):
    mood_prediction = rf_model.predict(text_input)
    return pd.DataFrame({"Emotion": mood_prediction}, index=[0])

# Song recommendation based on mood
def Recommend_Songs(text, df):
    preprocessed_text = preprocessed_data(text)
    text_vector = tfidf_vect.transform([preprocessed_text])
    predicted_emotion = predict_mood(rf_model, text_vector)["Emotion"].iloc[0]

    # if predicted_emotion == 'sadness':
    #     recommend = df[df['cluster'] == 0].sort_values(by="popularity", ascending=False).head(5)
    # elif predicted_emotion == 'love':
    #     recommend = df[df['cluster'] == 1].sort_values(by="popularity", ascending=False).head(5)
    # elif predicted_emotion in ['anger', 'fear']:
    #     recommend = df[df['cluster'] == 2].sort_values(by="popularity", ascending=False).head(5)
    # elif predicted_emotion in ['joy', 'surprise']:
    #     recommend = df[df['cluster'] == 3].sort_values(by="popularity", ascending=False).head(5)
    # else:
    #     recommend = pd.DataFrame()

    if predicted_emotion == 'sadness':
        recommend = df[df['cluster'] == 0].head(5)
    elif predicted_emotion == 'love':
        recommend = df[df['cluster'] == 1].head(5)
    elif predicted_emotion in ['anger', 'fear']:
        recommend = df[df['cluster'] == 2].head(5)
    elif predicted_emotion in ['joy', 'surprise']:
        recommend = df[df['cluster'] == 3].head(5)
    else:
        recommend = pd.DataFrame()

    recommend['YouTube Link'] = recommend['name'].apply(lambda x: f"https://www.youtube.com/results?search_query={x.replace(' ', '+')}")
    recommend['Spotify Link'] = recommend['name'].apply(lambda x: f"https://open.spotify.com/search/{x.replace(' ', '%20')}")
    # youtube_url = recommend['YouTube Link']
    # st.write("check out this [link](%s)\n" % youtube_url[1] + "\n")
    # st.write("check out this [link](%s)\n" % youtube_url[2] + "\n")
    return recommend[['name', 'artists', 'YouTube Link', 'Spotify Link']]

# Streamlit application
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Welcome", "Mood Music Recommender", "Similar Songs Recommender", "Visualization"])

def welcome_page():
    st.title("\U0001F3B5 Welcome to Music Recommendation System \U0001F3B6")
    st.markdown("""### Discover music that matches your mood or find similar songs easily. \U0001F3A7✨""")
    st.image("https://via.placeholder.com/800x400", caption="Music for every emotion", use_container_width=True)

def mood_recommender():
    st.title("\U0001F3BC Mood Music Recommender")
    user_input = st.text_input("How are you feeling today?")
    if st.button("Predict and Recommend"):
        if user_input:
            st.write("Here are the top 5 songs that suit your current mood the best. Enjoy!")
            recommendations = Recommend_Songs(user_input, data)
            if not recommendations.empty:
                st.dataframe(recommendations)
            else:
                st.write("No recommendations available for the predicted emotion.")

def similar_songs():
    st.title("\U0001F3B6 Similar Songs Recommender")
    st.write("Available Songs in Database:")
    st.dataframe(data[['name', 'artists']], use_container_width=True)
    song_name = st.text_input("Enter a song name:")
    if st.button("Find Similar Songs"):
        try:
            similar = get_similar_songs(song_name)
            st.subheader("🎵 Songs similar to your input:")
            st.dataframe(similar)
        except Exception as e:
            st.error(str(e))

def visualization_page():
    st.title("\U0001F4CA Visualization")
    if st.button("Show Elbow Method"):
        ssd = []
        for k in range(1, 11):
            kmeans = KMeans(n_clusters=k, random_state=0).fit(data.iloc[:, :-1])
            ssd.append(kmeans.inertia_)
        plt.figure(figsize=(8, 6))
        plt.plot(range(1, 11), ssd, marker='o', color='skyblue')
        plt.xlabel('Number of clusters', fontsize=12)
        plt.ylabel('Sum of squared distances', fontsize=12)
        plt.title('Elbow method', fontsize=16, color='darkblue')
        plt.grid(color='lightgray', linestyle='--', linewidth=0.5)
        st.pyplot(plt)

if page == "Welcome":
    welcome_page()
elif page == "Mood Music Recommender":
    mood_recommender()
elif page == "Similar Songs Recommender":
    similar_songs()
elif page == "Visualization":
    visualization_page()
