# Hate Speech Detection using Machine Learning

# Importing libraries
import pandas as pd
import numpy as np
import re
import nltk
import string

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

import seaborn as sns
import matplotlib.pyplot as plt


# Download stopwords (only first time)

nltk.download('stopwords')
# Load Dataset
dataset = pd.read_csv("twitter.csv")

print("\nFirst 5 Rows:")
print(dataset.head())

print("\nMissing Values:")
print(dataset.isnull().sum())

print("\nDataset Information:")
print(dataset.info())

print("\nDataset Description:")
print(dataset.describe())

# Mapping Labels
dataset["labels"] = dataset["class"].map({
    0: "hate speech",
    1: "offensive language",
    2: "no hate or offensive language"
})

data = dataset[["tweet", "labels"]]

# Text Cleaning
stop_words = set(stopwords.words("english"))
stemmer = SnowballStemmer("english")


def clean_data(text):
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'\{.*?\}', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r"[%s]" % re.escape(string.punctuation), "", text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\w*\d\w*', '', text)

    text = [word for word in text.split()
            if word not in stop_words]

    text = " ".join(text)

    text = [stemmer.stem(word)
            for word in text.split()]

    text = " ".join(text)

    return text


# Clean all tweets
data["tweet"] = data["tweet"].apply(clean_data)

# Feature Extraction
X = np.array(data["tweet"])
Y = np.array(data["labels"])

cv = CountVectorizer()
X = cv.fit_transform(X)

# Train Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    Y,
    test_size=0.33,
    random_state=42
)


# Model Training
dt = DecisionTreeClassifier()

dt.fit(X_train, y_train)

# Prediction and Accuracy
y_pred = dt.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

print("\nAccuracy :")
print(accuracy_score(y_test, y_pred))


# Confusion Matrix

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# Testing Custom Sentences


while True:

    sample = input("\nEnter a sentence (or type exit): ")

    if sample.lower() == "exit":
        break

    sample = clean_data(sample)

    data_test = cv.transform([sample]).toarray()

    prediction = dt.predict(data_test)

    print("Prediction :", prediction[0])

print("\nProgram Finished.")