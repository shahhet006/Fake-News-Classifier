# Fake News Detection Using Machine Learning

## Description
This project implements a machine learning model to detect fake news articles using Natural Language Processing (NLP) techniques. The system analyzes article titles using text preprocessing and the Multinomial Naive Bayes classifier to distinguish between fake and real news.

## Table of Contents
- [Installation](#installation)
- [Features](#features)
- [Usage](#usage)
- [Model Details](#model-details)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Installation

### Prerequisites
```bash
pip install -r requirements.txt
```

### Requirements
```
scikit-learn
numpy
matplotlib
nltk
pandas
```

## Features
* Text preprocessing using Porter Stemming
* Stop words removal
* Multinomial Naive Bayes classification
* Confusion matrix visualization
* Accuracy metrics reporting

## Usage

### Data Preprocessing
```python
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
import re

ps = PorterStemmer()
corpus = []
for i in range(0, len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['title'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)
```

### Training and Prediction
```python
from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(X_train, y_train)
pred = classifier.predict(X_test)
```

## Model Details

### Preprocessing Steps
1. Special character removal using regex
2. Conversion to lowercase
3. Porter Stemming implementation
4. Stop words removal

### Machine Learning Model
* **Algorithm**: Multinomial Naive Bayes
* **Input**: Processed article titles
* **Output**: Binary classification (FAKE/REAL)
* **Visualization**: Confusion matrix with customizable normalization

## Results

### Model Performance Metrics
The system evaluates performance using:
* Accuracy score
* Detailed confusion matrix showing:
  * True Positives (Correctly identified real news)
  * True Negatives (Correctly identified fake news)
  * False Positives (Fake news classified as real)
  * False Negatives (Real news classified as fake)

### Visualization
![Confusion Matrix](assets/confusion_matrix.png)

## Contributing
1. Fork the repository
2. Create a new branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Contact
Your Name - your.email@example.com
Project Link: [https://github.com/yourusername/fake-news-detection](https://github.com/yourusername/fake-news-detection)
