import streamlit as st
import joblib
from sklearn.feature_extraction.text import CountVectorizer

# Load the saved model and vectorizer
model = joblib.load("naive_bayes_model.pkl")
vectorizer = joblib.load("count_vectorizer.pkl")  # Save your CountVectorizer when saving the model

# Streamlit App Title
st.title("Spam Email Detection App")

email_text = st.text_area("Email Text", "")

# Predict Button
if st.button("PREDICT"):
    if email_text.strip():  # Ensure the input is not empty
        # Transform the input text using the vectorizer
        email_vectorized = vectorizer.transform([email_text])
        
        # Make the prediction
        prediction = model.predict(email_vectorized)[0]
        prediction_label = "Spam" if prediction == 1 else "Ham"
        
        # Display the result
        if prediction_label == "Spam":
            st.error("🚨 This email is classified as: **Spam**")
        else:
            st.success("✅ This email is classified as: **Ham**")
    else:
        st.warning("Please enter an email to predict.")

# Sidebar Information
st.sidebar.title("App Info")
st.sidebar.write("This app detects whether an email is Spam or Ham using a Naive Bayes model.")
st.sidebar.write("The model is trained on a spam email dataset.")
