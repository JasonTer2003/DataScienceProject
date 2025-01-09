import streamlit as st
import pandas as pd
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load trained models
with open("rf_model.pkl", "rb") as file:
    rf_model = joblib.load(file)

with open("tfidf_parameters.pkl", "rb") as file:
    tfidf_vect = joblib.load(file)

# Cache data loading
@st.cache_data
def load_data():
    df = pd.read_csv(r'C:\Drive D\UM\Y3S1\WIH3001 DSP\Latest Project\df_for_recommender.csv')
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

# Mood prediction function
def predict_mood(rf_model, text_input):
    mood_prediction = rf_model.predict(text_input)
    return pd.DataFrame({"Emotion": mood_prediction}, index=[0])

# Song recommendation based on mood
def Recommend_Songs(text, df):
    preprocessed_text = preprocessed_data(text)
    text_vector = tfidf_vect.transform([preprocessed_text])
    predicted_emotion = predict_mood(rf_model, text_vector)["Emotion"].iloc[0]

    if predicted_emotion == 'sadness':
        recommend = df[df['cluster'] == 1].head(5)
    elif predicted_emotion == 'love':
        recommend = df[df['cluster'] == 2].head(5)
    elif predicted_emotion in ['anger', 'fear']:
        recommend = df[df['cluster'] == 0].head(5)
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


st.title("\U0001F3BC Mood Music Recommender")
user_input = st.text_input("How are you feeling today?")
if st.button("Predict and Recommend"):
    if user_input:
        st.write("Here are the top 5 songs that suit your current mood the best. Enjoy!")
        recommendations = Recommend_Songs(user_input, data)
        if not recommendations.empty:
            st.dataframe(
                recommendations,
                column_config={
                    "YouTube Link": st.column_config.LinkColumn("YouTube URL", display_text="YouTube Link"),
                    "Spotify Link": st.column_config.LinkColumn("Spotify URL", display_text="Spotify Link"),
                },
                hide_index=True,
                use_container_width=True
            )
        else:
            st.write("No recommendations available for the predicted emotion.")