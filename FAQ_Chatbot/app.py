import streamlit as st
import pandas as pd
import nltk
import string

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download stopwords
nltk.download('stopwords')

# Load FAQ data
faq = pd.read_csv("faq.csv")

# Text preprocessing
stop_words = set(stopwords.words("english"))

def preprocess(text):
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    words = text.split()

    # Remove stopwords
    words = [word for word in words if word not in stop_words]

    return " ".join(words)

faq["Processed"] = faq["Question"].apply(preprocess)

# Convert questions into vectors
vectorizer = TfidfVectorizer()

faq_vectors = vectorizer.fit_transform(faq["Processed"])

# Streamlit UI
st.set_page_config(page_title="FAQ Chatbot", page_icon="🤖")

st.title("🤖 FAQ Chatbot")

st.write("Ask any question from the FAQ database.")

user_question = st.text_input("Your Question")

if st.button("Get Answer"):

    if user_question.strip() == "":
        st.warning("Please enter a question.")

    else:

        processed_question = preprocess(user_question)

        user_vector = vectorizer.transform([processed_question])

        similarity = cosine_similarity(user_vector, faq_vectors)

        best_match = similarity.argmax()

        score = similarity[0][best_match]

        if score >= 0.25:
            st.success(faq.iloc[best_match]["Answer"])
        else:
            st.error("Sorry! I couldn't find a matching answer.")