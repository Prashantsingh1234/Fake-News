import os
import pickle
import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import streamlit as st

# Download NLTK stopwords
nltk.download('stopwords')

# Define a stemming function for text preprocessing
port_stem = PorterStemmer()
def stemming(content):
    # Remove non-alphabetical characters, lower case, and split the text
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower().split()
    # Remove stopwords and stem the remaining words
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if word not in stopwords.words('english')]
    return ' '.join(stemmed_content)

# Define file paths for saving/loading the model and vectorizer
model_file = "model.pkl"
vectorizer_file = "vectorizer.pkl"

# Check if model and vectorizer are already saved. If yes, load them.
if os.path.exists(model_file) and os.path.exists(vectorizer_file):
    with open(model_file, "rb") as f:
        model = pickle.load(f)
    with open(vectorizer_file, "rb") as f:
        vectorizer = pickle.load(f)
else:
    # If not saved, load the dataset and train the model
    news_dataset = pd.read_csv(r"train.csv")
    news_dataset = news_dataset.fillna('')
    # Combine author and title into one text column
    news_dataset['content'] = news_dataset['author'] + ' ' + news_dataset['title']
    # Apply text preprocessing (stemming)
    news_dataset['content'] = news_dataset['content'].apply(stemming)
    
    # Separate the features and labels
    X = news_dataset['content'].values
    Y = news_dataset['label'].values
    
    # Convert text to numerical features using TF-IDF
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(X)
    
    # Split the dataset into training and testing sets (used here only for training)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
    
    # Train the Logistic Regression model
    model = LogisticRegression()
    model.fit(X_train, Y_train)
    
    # Save the trained model and vectorizer to disk
    with open(model_file, "wb") as f:
         pickle.dump(model, f)
    with open(vectorizer_file, "wb") as f:
         pickle.dump(vectorizer, f)

# ----------------- Streamlit UI -----------------
st.title("📰 Fake News Detection System")
st.subheader("Enter the news headline and author name to predict its authenticity")

# Take user input
author_input = st.text_input("Author Name")
title_input = st.text_area("News Title")

if st.button("Predict"):
    if author_input and title_input:
        # Combine the input fields and preprocess the text
        user_input = author_input + " " + title_input
        user_input_processed = stemming(user_input)
        user_input_vectorized = vectorizer.transform([user_input_processed])
        
        # Make prediction using the saved (or pre-trained) model
        prediction = model.predict(user_input_vectorized)
        if prediction[0] == 0:
            st.success("✅ The news is **Real**")
        else:
            st.error("❌ The news is **Fake**")
    else:
        st.warning("⚠️ Please enter both Author Name and News Title.")
