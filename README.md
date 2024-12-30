#Fake News Detection Using Machine Learning

#Overview
This project implements a machine learning model to detect fake news articles using Natural Language Processing (NLP) techniques. The system analyzes article titles using text preprocessing and the Multinomial Naive Bayes classifier to distinguish between fake and real news.

#Features

-Text preprocessing using Porter Stemming
-Stop words removal
-Multinomial Naive Bayes classification
-Confusion matrix visualization
-Accuracy metrics reporting

The system implements several preprocessing steps to clean and standardize the text data:

-Special character removal using regex
-Conversion to lowercase
-Porter Stemming implementation
-Stop words removal

Machine Learning Model

-Algorithm: Multinomial Naive Bayes
-Input: Processed article titles
-Output: Binary classification (FAKE/REAL)
-Visualization: Confusion matrix with customizable normalization

Dependencies

-scikit-learn
-numpy
-matplotlib
-NLTK


#Model Performance
The system evaluates performance using:

Accuracy score
Detailed confusion matrix showing:

-True Positives (Correctly identified real news)
-True Negatives (Correctly identified fake news)
-False Positives (Fake news classified as real)
-False Negatives (Real news classified as fake)


