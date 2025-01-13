Import necessary libraries
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

"""2. Understanding Dataset"""

# Read datasets
text_df_train = pd.read_csv(r'C:\Drive D\UM\Y3S1\WIH3001 DSP\Mood Detection\train.txt',
                      delimiter=';', names=['Text','Emotion'])
text_df_val = pd.read_csv(r'C:\Drive D\UM\Y3S1\WIH3001 DSP\Mood Detection\val.txt',
                      delimiter=';', names=['Text','Emotion'])
text_df_test = pd.read_csv(r'C:\Drive D\UM\Y3S1\WIH3001 DSP\Mood Detection\test.txt',
                      delimiter=';', names=['Text','Emotion'])

"""Assessing Datasets"""

# Training Dataset
# Print the first 5 rows of Training Dataset
print('First 5 rows of Training Datset: ', text_df_train.head())

# Print the shape of the Training Dataset
print('\nThe shape of Training Datset: ', text_df_train.shape)

# Validation Dataset
# Print the first 5 rows of Validation Dataset
print('First 5 rows of Validation Datset: ', text_df_val.head())

# Print the shape of the Validation Dataset
print('\nThe shape of Validation Datset: ', text_df_val.shape)

# Testing Dataset
# Print the first 5 rows of Testing Dataset
print('First 5 rows of Testing Datset: ', text_df_test.head())

# Print the shape of the Testing Dataset
print('\nThe shape of Testing Datset: ', text_df_test.shape)

"""Preprocessing on Training Dataset"""

# Check if the data is balanced or not
print(text_df_train.Emotion.value_counts())

# Check if the data is balanced or not (in percentage form)
text_df_train.Emotion.value_counts() / text_df_train.shape[0] *100

# Count Emotion distributions using a Pie Chart
emotion_counts = text_df_train['Emotion'].value_counts()
colors = sns.husl_palette(n_colors=len(emotion_counts))
plt.figure(figsize=(8, 8))
plt.pie(emotion_counts, labels=emotion_counts.index, autopct='%1.1f%%', startangle=140, colors = colors)
plt.title('Emotion Distribution on Training Dataset')
plt.show()

# Plotting a Count Plot on Training Datset
plt.figure(figsize=(8,4))
plt.title('Emotion Distribution on Training Dataset')
sns.countplot(x='Emotion', data=text_df_train, hue='Emotion');

"""Although the data is imbalanced and we may try to balance it out, however, all emotions that we interested in are the most repeated emotion so, till now we can keep the data as it is"""

# Remove data whose Emotion is 'love' (if necessary)
# text_df_train = text_df_train[~text_df_train['Emotion'].str.contains('love')]
# text_df_val = text_df_val[~text_df_val['Emotion'].str.contains('love')]
# text_df_test = text_df_test[~text_df_test['Emotion'].str.contains('love')]

# Check if there is any null values in each column
text_df_train.isnull().sum()

# Check if there is any duplicated values
text_df_train.duplicated().sum()

# Remove duplicated values
index = text_df_train[text_df_train.duplicated() == True].index
text_df_train.drop(index, axis = 0, inplace = True)
text_df_train.reset_index(inplace=True, drop = True)

# Check if there are any rows which are duplicated in text but with different emotions
print(text_df_train[text_df_train['Text'].duplicated() == True])
text_df_train['Text'].duplicated().sum()

# Print one of the example row to check if it is true
text_df_train[text_df_train['Text'] == text_df_train.iloc[6563]['Text']]

# Remove duplicated text
index = text_df_train[text_df_train['Text'].duplicated() == True].index
text_df_train.drop(index, axis = 0, inplace = True)
text_df_train.reset_index(inplace=True, drop = True)

# Check again if they have been removed
text_df_train[text_df_train['Text'].duplicated() == True]

# Stopwords from mltk library
import nltk
from nltk.corpus import stopwords
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# Count the number of stopwords in the data
temp = text_df_train.copy()
temp['stop_words'] = temp['Text'].apply(lambda x: len(set(x.split()) & set(stop_words)))
temp.stop_words.value_counts()

"""The data contains a lot of stopwords (more than 25 stopwaords, with the highest reaching 29 stopwords)."""

# Visualize the distribution of stopwords
sns.set(font_scale=1.3)
temp['stop_words'].plot(kind= 'hist', xlabel = 'Number of stopwwords per row in Training Dataset')

"""Preprocessing on Validation Dataset"""

# Check if the data is balanced or not
text_df_val.Emotion.value_counts()

# Check if the data is balanced or not (in percentage form)
text_df_val.Emotion.value_counts() / text_df_val.shape[0] *100

# Count Emotion distributions using a Pie Chart on Validation Dataset
emotion_counts = text_df_val['Emotion'].value_counts()
colors = sns.husl_palette(n_colors=len(emotion_counts))
plt.figure(figsize=(8, 8))
plt.pie(emotion_counts, labels=emotion_counts.index, autopct='%1.1f%%', startangle=140, colors = colors)
plt.title('Emotion Distribution on Validation Dataset')
plt.show()

# Plotting a count plot based on Validation Dataset
plt.figure(figsize=(8,4))
sns.countplot(x='Emotion', data=text_df_val, hue='Emotion')

# Check if there is any null values in each column
text_df_val.isnull().sum()

# Check if there is any duplicated values
text_df_val.duplicated().sum()

# Check if there are any rows which are duplicated in text but with differnet emotions
text_df_val[text_df_val['Text'].duplicated() == True]

# Testing on one of the examples
text_df_val[text_df_val['Text'] == text_df_val.iloc[1993]['Text']]

# Remove duplicated text
index = text_df_val[text_df_val['Text'].duplicated() == True].index
text_df_val.drop(index, axis = 0, inplace = True)
text_df_val.reset_index(inplace=True, drop = True)

# Check again if they have been removed
text_df_val[text_df_val['Text'].duplicated() == True]

# Count the number of stopwords in the validation dataset
temp_val = text_df_val.copy()
temp_val['stop_words'] = temp['Text'].apply(lambda x: len(set(x.split()) & set(stop_words)))
temp_val.stop_words.value_counts()

# Visualize the distribution of stopwords
sns.set(font_scale=1.3)
temp_val['stop_words'].plot(kind= 'hist', xlabel = 'Number of stopwwords per row in Validation Dataset')

"""Preprocessing on Testing Dataset"""

# Check if the data is balanced or not
text_df_test.Emotion.value_counts()

# Check if the data is balanced or not
text_df_test.Emotion.value_counts() / text_df_test.shape[0] *100

# Count Emotion distributions using a Pie Chart
emotion_counts = text_df_test['Emotion'].value_counts()
colors = sns.husl_palette(n_colors=len(emotion_counts))
plt.figure(figsize=(8, 8))
plt.pie(emotion_counts, labels=emotion_counts.index, autopct='%1.1f%%', startangle=140, colors = colors)
plt.title('Emotion Distribution on Testing Dataset')
plt.show()

# Plotting a Count Plot on Testing Datset
plt.figure(figsize=(8,4))
plt.title('Emotion Distribution on Testing Dataset')
sns.countplot(x='Emotion', data=text_df_test, hue='Emotion')

# Check if there is any null values in each column
text_df_test.isnull().sum()

# Check if there is any duplicated values
text_df_test.duplicated().sum()

# Check if there are any rows which are duplicated in text but with different emotions
text_df_test[text_df_test['Text'].duplicated() == True]

# Count the number of stopwords in the test dataset
temp_test = text_df_test.copy()
temp_test['stop_words'] = temp['Text'].apply(lambda x: len(set(x.split()) & set(stop_words)))
temp_test.stop_words.value_counts()

# Visualize the distribution of stopwords in test dataset
sns.set(font_scale=1.3)
temp_test['stop_words'].plot(kind= 'hist', xlabel = 'Number of stopwords in Testing Dataset')

"""Cleaning Data"""

# Import necessary libraries
import re
from nltk.stem import WordNetLemmatizer

nltk.download('wordnet')  # Download WordNet for lemmatizer
nltk.download('omw-1.4')  # Download additional WordNet support files

def lemmatization(text):
    lemmatizer= WordNetLemmatizer()

    text = text.split()

    text=[lemmatizer.lemmatize(y) for y in text]

    return " " .join(text)

def remove_stop_words(text):

    Text=[i for i in str(text).split() if i not in stop_words]
    return " ".join(Text)

def Removing_numbers(text):
    text=''.join([i for i in text if not i.isdigit()])
    return text

def lower_case(text):

    text = text.split()

    text=[y.lower() for y in text]

    return " " .join(text)

def Removing_punctuations(text):
    ## Remove punctuations
    text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), ' ', text)
    text = text.replace('؛',"", )

    ## remove extra whitespace
    text = re.sub('\s+', ' ', text)
    text =  " ".join(text.split())
    return text.strip()

def Removing_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def remove_small_sentences(text_df):
    for i in range(len(text_df)):
        if len(text_df.text.iloc[i].split()) < 3:
            text_df.text.iloc[i] = np.nan

def normalize_text(text_df):
    text_df.Text=text_df.Text.apply(lambda text : lower_case(text))
    text_df.Text=text_df.Text.apply(lambda text : remove_stop_words(text))
    text_df.Text=text_df.Text.apply(lambda text : Removing_numbers(text))
    text_df.Text=text_df.Text.apply(lambda text : Removing_punctuations(text))
    text_df.Text=text_df.Text.apply(lambda text : Removing_urls(text))
    text_df.Text=text_df.Text.apply(lambda text : lemmatization(text))
    return text_df

def normalized_sentence(sentence):
    sentence= lower_case(sentence)
    sentence= remove_stop_words(sentence)
    sentence= Removing_numbers(sentence)
    sentence= Removing_punctuations(sentence)
    sentence= Removing_urls(sentence)
    sentence= lemmatization(sentence)
    return sentence

# Example to test if the segment of code (removing stopwords) works
normalized_sentence("My Name is Jason, and I feel very happy because today is Sunday_")

# Normalize all the texts following functions above based on nltk
text_df_train= normalize_text(text_df_train)
text_df_val= normalize_text(text_df_val)
text_df_test= normalize_text(text_df_test)

"""Modelling"""

# Divide to preprocess texts
text_X_train = text_df_train['Text'].values
text_y_train = text_df_train['Emotion'].values

text_X_val = text_df_val['Text'].values
text_y_val = text_df_val['Emotion'].values

text_X_test = text_df_test['Text'].values
text_y_test = text_df_test['Emotion'].values

"""TF-DIF and Pipeline Object are used for most of the machine learning models as below.

Explanation on TF-DIF and Pipeline Object:
1. TF-IDF Vectorization
- The TfidfVectorizer is used to convert raw text into numerical feature vectors.
- TF-IDF assigns weights to words based on their frequency in a document (Term Frequency, TF) and how rare they are across all documents (Inverse Document Frequency, IDF).
- Words frequent in a single document get a higher weight.
- Words common across many documents (like stop words) get a lower weight.
2. Pipeline
- The function uses Pipeline from sklearn.pipeline to create a streamlined process that:
- Transforms the input text using TF-IDF ('vect').
- Applies the specified model ('clf') on the transformed text.
3. Training
- The fit method is called to train the pipeline:
- First, the text is converted to numerical TF-IDF vectors.
- Then, the model is trained using these vectors and their corresponding targets.

Workflow
1. Input Data: Raw text data (data) and their corresponding labels (targets).

2. Vectorization:
- Converts text into numerical representations using TF-IDF.
- Ensures the model can work with text data.
3. Model Training:
- Trains the specified model using the TF-IDF-transformed data.
4. Output:
- Returns a trained pipeline (text_clf) that combines the TF-IDF vectorizer and the model.
"""

# Import libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

# Import libraries for all models
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score

def train_model(model, data, targets):
    """
    To train a model on the given data and targets

    Parameters:
    1. model (sklearn model): The model to be trained
    2. data (list of str): The input data
    3. targets (list of str): The targets

    Returns:
    Pipeline: The trained model as a Pipeline object.
    """
    # Create a Pipeline object with a TfidfVectorizer and the given model
    text_clf = Pipeline([('vect',TfidfVectorizer()),
                         ('clf', model)])

    # Fit the model on the data and targets
    text_clf.fit(data, targets)
    return text_clf

"""F1 score refers to the harmonic mean (a kind of average) of precision and recall.
This metric balances the importance of precision and recall, which is important for our class-imbalanced datasets.
"""

def get_F1(trained_model, text_X, text_y):
    """
    Get the F1 score for the given model on the given data and targets.

    Parameters:
    trained_model (sklearn model): The trained model.
    X (list of str): The input data.
    y (list of str): The targets.

    Returns:
    array: The F1 score for each class.
    """
    # Make predictions on the input data using the trained model
    predicted = trained_model.predict(text_X)

    # Calculate the F1 score for the predictions
    f1 = f1_score(text_y, predicted, average = None)

    # Return the F1 score
    return f1

"""Logistic Regression"""

# Import libraries
from sklearn.linear_model import LogisticRegression

# Train the model with the training data
log_reg = train_model(LogisticRegression(solver ='liblinear',random_state = 0), text_X_train, text_y_train)

# Test the model with the test data
text_y_pred = log_reg.predict(text_X_test)

# Calculate accuracy
log_reg_accuracy = accuracy_score(text_y_test, text_y_pred)
print('Logistic Regression Accuracy: ', log_reg_accuracy)

# Calculate F1 score
f1_Score = get_F1(log_reg, text_X_test, text_y_test)
pd.DataFrame(f1_Score, index = text_df_train.Emotion.unique(), columns = ['F1 score from Logistic Regression'])

# Classification Report
print(classification_report(text_y_test, text_y_pred))

"""Decision Tree"""

# Import library
from sklearn.tree import DecisionTreeClassifier

# Train the model with the training data
dt = train_model(DecisionTreeClassifier(random_state = 0), text_X_train, text_y_train)

#test the model with the test data
text_y_pred = dt.predict(text_X_test)

# Calculate the Decision Tree accuracy
dt_accuracy = accuracy_score(text_y_test, text_y_pred)
print('Decision Tree Accuracy: ', dt_accuracy)

# Calculate the F1 score
f1_Score = get_F1(dt, text_X_test, text_y_test)
pd.DataFrame(f1_Score, index = text_df_train.Emotion.unique(), columns=['F1 score'])

# Classification Report for Decision Tree
print(classification_report(text_y_test, text_y_pred))

"""Support Vector Machine"""

# Import library
from sklearn.svm import SVC

# Train the model with the training data
svm = train_model(SVC(random_state = 0), text_X_train, text_y_train)

# Test the model with the test data
text_y_pred = svm.predict(text_X_test)

# Calculate the accuracy
svm_accuracy = accuracy_score(text_y_test, text_y_pred)
print('Support Vector Machine Accuracy: ', svm_accuracy)

# Calculate the F1 score
f1_Score = get_F1(svm, text_X_test, text_y_test)
pd.DataFrame(f1_Score, index = text_df_train.Emotion.unique(), columns=['F1 score'])

# Classification Report for Support Vector Machine
print(classification_report(text_y_test, text_y_pred))

"""Random Forest"""

# Import library
from sklearn.ensemble import RandomForestClassifier

# Train the model with the training data
rf = train_model(RandomForestClassifier(random_state = 0), text_X_train, text_y_train)

# Test the model with the test data
text_y_pred = rf.predict(text_X_test)

# Calculate the accuracy
rf_accuracy = accuracy_score(text_y_test, text_y_pred)
print('Random Forest Accuracy: ', rf_accuracy)

# Calculate the F1 score
f1_Score = get_F1(rf, text_X_test, text_y_test)
pd.DataFrame(f1_Score, index = text_df_train.Emotion.unique(), columns=['F1 score'])

# Classification Report for Random Forest
print(classification_report(text_y_test, text_y_pred))

"""Results of All Models"""

models = pd.DataFrame({
    'Model': ['Logistic Regression', 'Decision Tree', 'Support Vector Machine', 'Random Forest'],
    'Accuracy': [round(log_reg_accuracy, 3), round(dt_accuracy, 3), round(svm_accuracy, 3), round(rf_accuracy, 3)]})

models.sort_values(by='Accuracy', ascending=False).reset_index().drop(['index'], axis=1)

# Import library
from sklearn.ensemble import RandomForestClassifier

def rf_model():
    # Train the model with the training data
    rf = train_model(RandomForestClassifier(random_state = 0), text_X_train, text_y_train)

    # Test the model with the test data
    text_y_pred = rf.predict(text_X_test)

    return text_y_pred

import json
import joblib
# Extract the TfidfVectorizer from the pipeline
tfidf_vect = rf.named_steps['vect']

# Filter out non-serializable parameters
def serialize_params(params):
    return {key: value for key, value in params.items() if isinstance(value, (str, int, float, bool, list, dict))}

tfidf_params = serialize_params(tfidf_vect.get_params())

with open("tfidf_parameters.pkl", "wb") as file:
    joblib.dump(tfidf_vect, file)

print("Filtered TfidfVectorizer parameters saved successfully.")

import json
from sklearn.ensemble import RandomForestClassifier
import joblib

# Extract the Random Forest classifier from the pipeline
rf_model = rf.named_steps['clf']

# Save model parameters and structure
rf_parameters = {
    "n_estimators": rf_model.n_estimators,
    "max_depth": rf_model.max_depth,
    "random_state": rf_model.random_state,
    "min_samples_split": rf_model.min_samples_split,
    "min_samples_leaf": rf_model.min_samples_leaf,
    "max_features": rf_model.max_features,
    "class_weight": rf_model.class_weight,
    "criterion": rf_model.criterion,
}

with open("rf_model.pkl", "wb") as file:
    joblib.dump(rf_model, file)

print("Random Forest model saved successfully as a .pkl file.")

# # text_mood/rf_model.py
# import pandas as pd

# def predict_mood(rf_model, text_input):
#     # Assuming text_input is preprocessed as needed
#     mood_prediction = rf_model.predict(text_input)
#     return pd.DataFrame({"Emotion": mood_prediction}, index=[0])

"""Image Data"""

# # display some images for every different expression

# import numpy as np
# import seaborn as sns
# from keras.preprocessing.image import load_img, img_to_array
# import matplotlib.pyplot as plt
# import os

# # size of the image: 48*48 pixels
# pic_size = 48

# base_path = r"C:\Drive D\UM\Y3S1\WIH3001 DSP\Image\\"

# plt.figure(0, figsize=(12,20))
# cpt = 0

# for expression in os.listdir(base_path + "train"):
#     for i in range(1,6):
#         cpt = cpt + 1
#         plt.subplot(7,5,cpt)
#         img = load_img(base_path + "train/" + expression + "/" +os.listdir(base_path + "train/" + expression)[i], target_size=(pic_size, pic_size))
#         plt.imshow(img, cmap="gray")

# plt.tight_layout()
# plt.show()

import kagglehub

# Download latest version
path = kagglehub.dataset_download("asaniczka/top-spotify-songs-in-73-countries-daily-updated")

print("Path to dataset files:", path)

import pandas as pd
actual_path = path + r"\universal_top_spotify_songs.csv"
df = pd.read_csv(actual_path)

# Understand number of rows and number of columns
print(df.shape)

# Understand the name, count and data type of each columns
df.info()

# import pandas as pd
# actual_path = r"C:\Users\Jason\Downloads\universal_top_spotify_songs.csv"
# df = pd.read_csv(actual_path)

# # Understand number of rows and number of columns
# print(df.shape)

# # Understand the name, count and data type of each columns
# df.info()

df.describe()

# Keep only rows where is_explicit is "FALSE" due to music therapy
df = df[df['is_explicit'] == False]

# Understand number of rows and number of columns
print(df.shape)

# Understand the name, count and data type of each columns
df.info()

# Print out the contents
df.head(20)

# df = df.dropna()

# # Understand number of rows and number of columns
# print(df.shape)

# # Understand the name, count and data type of each columns
# df.info()

# # Print out the contents
# df.head(20)

# Myanmar, Cambodia, Laos, East Timor are not in the dataset although they are South East Asian countries
selected_countries = ['BN', 'ID', 'MY', 'PH', 'SG', 'TH', 'VN']
df = df[df['country'].isin(selected_countries)]

# selected_countries = ['MY']
# df = df[df['country'].isin(selected_countries)]

# df = df[df['country'] == 'MY']

# Understand number of rows and number of columns
print(df.shape)

# Understand the name, count and data type of each columns
df.info()

# Print out the contents
df.head(20)

df['snapshot_date'] = pd.to_datetime(df['snapshot_date'])

song_info_features = ['name', 'artists', 'spotify_id', 'popularity', 'country']
song_features_normalized = ['valence', 'acousticness', 'danceability', 'energy', 'instrumentalness', 'liveness', 'speechiness']
song_features_not_normalized = ['duration_ms', 'key', 'loudness', 'mode', 'tempo']

from sklearn.preprocessing import StandardScaler
# define song features
# features = ['acousticness', 'danceability', 'energy', 'instrumentalness', 'key', 'liveness', 'loudness', 'speechiness', 'valence', 'tempo']
song_featues = song_features_normalized + song_features_not_normalized
# standardize the features
scaler = StandardScaler()
feature_for_cluster = scaler.fit_transform(df[song_featues])

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# make a correlation matrix of the song features
corr = np.corrcoef(feature_for_cluster.T)

# plot the correlation matrix
plt.figure(figsize=(12, 10))
sns.heatmap(corr, annot=True,
            xticklabels=song_featues, yticklabels=song_featues, cmap='Blues')
plt.title('Correlation matrix of song features')
plt.show()

from sklearn.decomposition import PCA

# apply dimensionality reduction to the standardized song features

# one way is to use PCA
pca = PCA(n_components=2)
feature_for_cluster_dim_redu = pca.fit_transform(feature_for_cluster)

# from sklearn.manifold import TSNE
# # another way is to use t-SNE
# tsne = TSNE(n_components=2, perplexity=30, random_state=0)
# feature_for_cluster_dim_redu = tsne.fit_transform(feature_for_cluster)

from sklearn.cluster import KMeans

# y_kmeans = kmeans.predict(songs_features)

# use the elbow method to find the optimal number of clusters
# calculate the sum of squared distances for different number of cluster
ssd = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=0).fit(feature_for_cluster_dim_redu)
    ssd.append(kmeans.inertia_)
# plot the sum of squared distances for different number of cluster
plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), ssd, marker='o')
plt.xticks(range(1, 11))
plt.xlabel('Number of clusters')
plt.ylabel('Sum of squared distances')
plt.title('Elbow method')
plt.show()

# cluster the features into 4 clusters using KMeans
kmeans = KMeans(n_clusters=4, random_state=0).fit(feature_for_cluster_dim_redu)
# add the cluster labels to the dataset
df['cluster'] = kmeans.labels_

# plot the clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(x=feature_for_cluster_dim_redu[:, 0], y=feature_for_cluster_dim_redu[:, 1],
                hue=df['cluster'], palette='Set2', size=df['popularity'], sizes=(10, 150), alpha=0.6)
# set legend outside the plot
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.title('Song features clustered into 4 groups')
plt.show()

df = df[df['country'] == 'MY']
df = df.drop_duplicates(subset=['name'])

# Cluster 0 songs
print(df[df['cluster'] == 0].shape)
df[df['cluster'] == 0].head(10)

# Cluster 1 songs
print(df[df['cluster'] == 1].shape)
df[df['cluster'] == 1].head(10)

# Cluster 2 songs
print(df[df['cluster'] == 2].shape)
df[df['cluster'] == 2].head(10)

# Cluster 3 songs
print(df[df['cluster'] == 3].shape)
df[df['cluster'] == 3].head(10)

# # Cluster 4 songs
# print(df[df['cluster'] == 4].shape)
# df[df['cluster'] == 4].head(10)

df.head(20)

"""High energy + high valence = Happy
Low energy + low valence = Sad

Cluster 0 = Sad
Cluster 1 = Love
Cluster 2 = Relaxed
Cluster 3 = Energetic
"""

# df['cluster'] = df['cluster'].map({
#     "Love": 2,
#     "Sad": 0,
#     "Relaxed": 1,
#     "Energetic": 3
# })

# df['cluster'] = df['cluster'].map({
#     0: "Sad",
#     1: "Relaxed",
#     2: "Love",
#     3: "Energetic"
# })

# df.head(20)

df.info()

"""Need to think of how to check clustering's accuracy"""

# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import cross_val_score
# from sklearn.metrics import classification_report, confusion_matrix
# from sklearn.utils.multiclass import unique_labels

# X = df
# y = df['cluster']

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

# rfc = RandomForestClassifier(n_estimators=100,criterion='gini')
# rfc.fit(X_train, y_train)

# def plot_confusion_matrix(y_true, y_pred, classes,
#                           normalize=False,
#                           title=None,
#                           cmap=plt.cm.Blues):
#     """
#     This function prints and plots the confusion matrix.
#     Normalization can be applied by setting `normalize=True`.
#     """
#     if not title:
#         if normalize:
#             title = 'Normalized confusion matrix'
#         else:
#             title = 'Confusion matrix'

#     # Compute confusion matrix
#     cm = confusion_matrix(y_true, y_pred)
#     # Only use the labels that appear in the data
#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#         print("Normalized confusion matrix")
#     else:
#         print('Confusion matrix, without normalization')

#     print(cm)

#     fig, ax = plt.subplots()
#     im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
#     ax.figure.colorbar(im, ax=ax)
#     # We want to show all ticks...
#     ax.set(xticks=np.arange(cm.shape[1]),
#            yticks=np.arange(cm.shape[0]),
#            # ... and label them with the respective list entries
#            xticklabels=classes, yticklabels=classes,
#            title=title,
#            ylabel='True label',
#            xlabel='Predicted label')

#     # Rotate the tick labels and set their alignment.
#     plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
#              rotation_mode="anchor")

#     # Loop over data dimensions and create text annotations.
#     fmt = '.2f' if normalize else 'd'
#     thresh = cm.max() / 2.
#     for i in range(cm.shape[0]):
#         for j in range(cm.shape[1]):
#             ax.text(j, i, format(cm[i, j], fmt),
#                     ha="center", va="center",
#                     color="white" if cm[i, j] > thresh else "black")
#     fig.tight_layout()
#     return ax

# # Confusion matrix
# definitions = ['Sad','Love','Relaxed','Energetic']
# # reversefactor = dict(zip(range(4),definitions))
# # actual = np.vectorize(reversefactor.get)(y_test)
# # pred = np.vectorize(reversefactor.get)(y_pred)
# # print(pd.crosstab(actual, pred, rownames=['Actual Mood'], colnames=['Predicted Mood']))

# plot_confusion_matrix(y_test, y_kmeans, classes=definitions,
#                       title='Confusion matrix for Random Forest')

# Check emotion and recommend based on predictions
# if text_y_pred[0] == 'Happy':  # Assuming text_y_pred is a list or array
recommend = df[df['cluster'] == 3].head(5)
recommend = recommend.sort_values(by="popularity", ascending=False)
print(recommend[['name', 'artists']])


# if(rf.text_y_pred['Emotion'] == 'Happy'):
    # recommend = df[df['cluster'] == 3].head(5)
    # recommend = recommend.sort_values(by="popularity", ascending=False)
    # print(recommend[['name', 'artists']])
#     display(recommend['name', 'artists'])

import re
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def preprocessed_data(text):
    # Define preprocessing functions
    def lemmatization(text):
        lemmatizer = WordNetLemmatizer()
        text = text.split()
        text = [lemmatizer.lemmatize(word) for word in text]
        return " ".join(text)

    def remove_stop_words(text):
        stop_words = set(stopwords.words('english'))
        text = [word for word in text.split() if word not in stop_words]
        return " ".join(text)

    def removing_numbers(text):
        return ''.join([char for char in text if not char.isdigit()])

    def lower_case(text):
        return text.lower()

    def removing_punctuations(text):
        # Remove punctuations and extra whitespace
        text = re.sub(r'[!"#$%&\'()*+,\-./:;<=>?@[\\\]^_`{|}~]', ' ', text)
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

import joblib
import pandas as pd

# Load the trained Random Forest model
with open("rf_model.pkl", "rb") as file:
    rf_model = joblib.load(file)

# Load the trained TfidfVectorizer model
with open("tfidf_parameters.pkl", "rb") as file:
    tfidf_vect = joblib.load(file)

def predict_mood(rf_model, text_input):
    """
    Predicts the mood/emotion from the input text.

    Args:
    rf_model: Trained Random Forest Classifier.
    text_input: Preprocessed text transformed using TfidfVectorizer.

    Returns:
    A DataFrame containing the predicted emotion.
    """
    # Predict the mood/emotion
    mood_prediction = rf_model.predict(text_input)
    return pd.DataFrame({"Emotion": mood_prediction}, index=[0])

def Recommend_Songs(text, df):
    """
    Recommends songs based on the emotion predicted from the input text.

    Args:
    text: Input text from the user.
    df: DataFrame containing song details and clusters.

    Returns:
    None: Prints the recommended songs.
    """
    # Preprocess and vectorize the input text
    preprocessed_text = preprocessed_data(text)
    text_vector = tfidf_vect.transform([preprocessed_text])

    # Predict the emotion
    predicted_emotion = predict_mood(rf_model, text_vector)["Emotion"].iloc[0]
    print(f"Predicted Emotion: {predicted_emotion}")

    # Recommend songs based on the predicted emotion
    print("Recommended Songs for Your Current Mood:")
    if predicted_emotion == 'sadness':
        recommend = df[df['cluster'] == 0].sort_values(by="popularity", ascending=False).head(5)
        print(recommend[['name', 'artists']])

    elif predicted_emotion == 'love':
        recommend = df[df['cluster'] == 1].sort_values(by="popularity", ascending=False).head(5)
        print(recommend[['name', 'artists']])

    elif predicted_emotion in ['anger', 'fear']:
        recommend = df[df['cluster'] == 2].sort_values(by="popularity", ascending=False).head(5)
        print(recommend[['name', 'artists']])

    elif predicted_emotion in ['joy', 'surprise']:
        recommend = df[df['cluster'] == 3].sort_values(by="popularity", ascending=False).head(5)
        print(recommend[['name', 'artists']])

    else:
        print("No recommendations available for the predicted emotion.")

# # Making Songs Recommendations Based on Predicted Class
# with open("rf_model.pkl", "rb") as file:
#     rf_model = joblib.load(file)

# # Load the trained TfidfVectorizer model
# with open("tfidf_vectorizer.pkl", "rb") as file:
#     tfidf_vect = joblib.load(file)

# def Recommend_Songs(text):

#     preprocessed_text = preprocessed_data(text)

#     rf_model.text

#     if(preprocessed_text['Emotion'] == 'Sadness'):
#         recommend = df[df['cluster'] == 0].head(5)
#         recommend = recommend.sort_values(by="popularity", ascending=False)
#         print(recommend[['name', 'artists']])
#         # display(recommend[['name', 'artists']])

#     if(text_y_pred['Emotion'] == 'Love'):
#         recommend = df[df['cluster'] == 1].head(5)
#         recommend = recommend.sort_values(by="popularity", ascending=False)
#         print(recommend[['name', 'artists']])

#     if(text_y_pred['Emotion'] == 'Anger' and 'Fear'):
#         recommend = df[df['cluster'] == 2].head(5)
#         recommend = recommend.sort_values(by="popularity", ascending=False)
#         print(recommend[['name', 'artists']])

#     if(text_y_pred['Emotion'] == 'Joy' and 'Surprise'):
#         recommend = df[df['cluster'] == 3].head(5)
#         recommend = recommend.sort_values(by="popularity", ascending=False)
#         print(recommend[['name', 'artists']])

"""Recommender"""

# build a recommendation system using the song features
# define the features to be used in the recommendation system

# features = song_features_normalized + song_features_not_normalized
features = song_features_normalized + song_features_not_normalized

# make a new dataframe, apply standardization to the features
scaler = StandardScaler()
feature_for_recommendation = scaler.fit_transform(df[features])

# create a dataframe as a copy of the original dataframe and with the standardized features
df_recommendation = df.copy()
df_recommendation[features] = feature_for_recommendation

# turn all song names into uppercase
df_recommendation['name'] = df_recommendation['name'].str.upper()

# save the dataframe
# df_recommendation.to_csv('data/processes/df_for_recommendation.csv', index=False)

df.head(20)

# # Remove songs whose popularity == 0
df = df[df['popularity'] != 0]

# Save the dataframe
df_recommendation.to_csv(r'C:\Drive D\UM\Y3S1\WIH3001 DSP\Latest Project\df_for_recommender.csv', index=False)

from sklearn.metrics.pairwise import cosine_similarity
from wordcloud import WordCloud
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

# Function to get the feature vector of a given song
def get_feature_vector(song_name):
    """
    Retrieves the feature vector for a given song by its name.
    
    Parameters:
    - song_name (str): Name of the song to search.
    - features (list): List of feature columns to include in the vector.

    Returns:
    - feature_vector (numpy.ndarray): Feature vector of the song.
    """
    df_song = df_recommendation.query('name == @song_name')

    if df_song.empty:
        raise Exception(
            "The song does not exist in the dataset! \nUse the search function if you are unsure of the song name."
        )

    feature_vector = df_song[features].values
    return feature_vector


# Function to find similar songs
def get_similar_songs(song_name, top_n):
    """
    Finds and recommends songs similar to the given song name.

    Parameters:
    - song_name (str): Name of the song to find similarities.
    - features (list): List of feature columns used for similarity.
    - top_n (int): Number of similar songs to return.
    - plot_type (str): Type of plot to display ('wordcloud' or 'bar').

    Returns:
    - similar_songs (DataFrame): DataFrame containing similar songs and their popularity.
    """
    feature_vector = get_feature_vector(song_name)

    # Calculate similarity
    similarities = cosine_similarity(df_recommendation[features].values, feature_vector).flatten()

    # Plot histogram of similarities
    sns.histplot(similarities, kde=True, bins=50)
    plt.title("Distribution of Cosine Similarity")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Frequency")
    plt.show()

    # Alternatively, use Euclidean distance
    # similarities = euclidean_distances(df_recommendation[features].values, feature_vector).flatten()

    # Get indices of top N+1 similar songs (excluding the input song itself)
    related_song_indices = similarities.argsort()[-(top_n + 1):][::-1][1:]

    # Select the top N similar songs and include their popularity
    similar_songs = df_recommendation.iloc[related_song_indices][['name', 'artists', 'popularity']]
    similar_songs = similar_songs.sort_values(by='popularity', ascending=False).reset_index(drop=True)

    return similar_songs

# Function to search for a song and display its information
def search_song(song_name):
    """
    Searches for a song by its name in the dataset.

    Parameters:
    - song_name (str): Name of the song to search.

    Returns:
    - DataFrame: DataFrame containing information about the song.
    """
    df_song = df_recommendation.query('name == @song_name')
    if df_song.empty:
        raise Exception("The song does not exist in the dataset!")
    
    print(f"Great! The following song(s) are in the dataset: {df_song[['name', 'artists']].to_numpy()}")
    return df_song[['name', 'artists', 'popularity']]


def display_available_songs():
    print("The top trending songs for today are:")
    return {df['name']}

display_available_songs

# # with the clusters, we find the top 5 popular genres in each cluster
# popular_genres = df.groupby('cluster').apply(lambda x: x.nlargest(5, columns=['popularity'])).reset_index()

# for i in range(5):
#     print(f"Cluster {i} : {popular_genres.query('cluster == @i')['genres'].tolist()}")

# jupyter nbconvert --to python similar_songs_recommender.ipynb
