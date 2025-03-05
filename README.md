# Twitter-Sentiment-Analysis Using Machine Learning
A machine learning project that classifies Twitter sentiments (Positive/Negative) using NLP techniques and Logistic Regression.

# Overview
This project analyzes tweets to determine sentiment, helping in opinion mining, brand monitoring, and social media analysis.
It processes text data, applies TF-IDF vectorization, and uses Logistic Regression to classify tweets.

# Technologies Used
->Python
->Pandas, NumPy – Data handling
->NLTK – Text preprocessing (stopword removal, stemming)
->Scikit-learn – ML model (Logistic Regression, TF-IDF)
->Google Colab – Model training & experimentation

# Dataset
-> The dataset used for training is **Sentiment140** from Kaggle.
-> Contains tweets labeled as positive (1) or negative (0). .  
-> 🔗 [Download Sentiment140 Dataset](https://www.kaggle.com/datasets/kazanova/sentiment140)

# Model Training & Workflow
Data Preprocessing:
Removed stopwords, special characters, URLs.
Applied stemming using PorterStemmer.

Feature Engineering:
Used TF-IDF Vectorizer to convert text into numerical form.

Model Selection:
Trained a Logistic Regression model for classification.
Evaluation Metrics:

Accuracy: 78.16%

# How to Run the Project
Open the notebook
Run all cells to train and test the model.

# Future Improvements
 Improve accuracy using Deep Learning (LSTMs, BERT).
 Handle sarcasm and negation better.
 Deploy the model as a web app using Flask/Streamlit.

# Connect With Me
 Email: nimeshmali2004@gmail.com
 
 LinkedIn:www.linkedin.com/in/nimesh-mali-4b590b2b3


