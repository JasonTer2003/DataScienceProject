import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from PIL import Image

# Cache data loading
@st.cache_data
def load_data():
    df = pd.read_csv(r'C:\Drive D\UM\Y3S1\WIH3001 DSP\Latest Project\df_for_recommender.csv')
    return df

data = load_data()

def listening_habits_dashboard():
    # Display static images
    img = Image.open(r"C:\Drive D\UM\Y3S1\WIH3001 DSP\Latest Project\listening_habits_dashboard.png")
    st.image(img, caption="Listening Habits of Malaysia Listeners")


def worldwide_dashboard():
    # Display static images
    img = Image.open(r"C:\Drive D\UM\Y3S1\WIH3001 DSP\Latest Project\worldwide_dashboard.png")
    st.image(img, caption="Analysis of Song Features Worldwide Dashboard")

st.title("\U0001F4CA Visualization")
st.markdown("Here is where you can understand music better! Click on the any following to have a peek.")
left, right = st.columns(2)
if left.button("Listening Habits of Malaysia Listeners Dashboard", icon="😃", use_container_width=True):
    listening_habits_dashboard()
if right.button("Analysis of Song Features Worldwide Dashboard", icon="😃", use_container_width=True):
    worldwide_dashboard()


