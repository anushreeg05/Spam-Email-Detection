# Spam-Email-Detection

This project demonstrates a machine learning pipeline to detect spam emails using a Naive Bayes classifier. It includes both a Jupyter Notebook for data preprocessing, model training, and evaluation, as well as a Streamlit-based web application for real-time spam detection.

Features
- Machine Learning Model: Utilizes a Naive Bayes classifier for spam email detection.
- Streamlit Web App: A user-friendly interface for entering email text and predicting if it's spam or not.
- Dataset: Trained on a labeled spam email dataset ('spam2.csv').


Repository Structure

📂 Spam Email Detection
- app.py               # Streamlit app script for the web interface
- spam_email.ipynb     # Jupyter Notebook for data analysis, preprocessing, and model training
- spam2.csv            # Dataset used for training and testing
- naive_bayes_model.pkl # Serialized Naive Bayes model (generated after training)
- count_vectorizer.pkl # Serialized CountVectorizer for text processing


Prerequisites
- Python 3.8+
- Required libraries:
  - 'streamlit'
  - 'scikit-learn'
  - 'joblib'
  - 'pandas'
  - 'numpy'


Usage

Model Training
1. Open the `spam_email.ipynb` file in Jupyter Notebook.
2. Run all cells to preprocess the data, train the Naive Bayes model, and save the model and vectorizer as `.pkl` files.

Web Application
1. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
2. Enter email text in the provided input box and click **PREDICT** to classify it as Spam or Ham.


Dataset
The dataset ('spam2.csv') includes labeled email samples categorized into:
- 'Ham' (not spam): Emails that are legitimate.
- 'Spam': Emails identified as unwanted or malicious.


Results
The trained model achieves high accuracy in distinguishing between spam and non-spam emails. Specific metrics (e.g., precision, recall, F1-score) can be found in the Jupyter Notebook.


Acknowledgements
- Libraries Used: Scikit-learn, Pandas, Streamlit, Joblib
- Dataset Source: [Include source if applicable]
  

Future Work
- Enhance the dataset with more diverse samples.
- Experiment with advanced models like Random Forest or XGBoost.
- Add email attachments analysis for more robust detection.
