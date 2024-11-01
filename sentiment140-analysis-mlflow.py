# Import MLflow
import mlflow
import mlflow.sklearn

# Set the experiment name
mlflow.set_experiment("Sentiment Analysis")

# Dataset
import kagglehub

# Utilities
import re
import time
import pickle
import numpy as np
import pandas as pd

# Plotting
import seaborn as sns
import matplotlib.pyplot as plt

# Nltk
import nltk
from nltk.stem import WordNetLemmatizer

# Sklearn
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report

# nltk.download('wordnet')

kazanova_sentiment140_path = kagglehub.dataset_download('kazanova/sentiment140')

print('Data source import complete.')

# Importing the dataset
DATASET_COLUMNS  = ["sentiment", "ids", "date", "flag", "user", "text"]
DATASET_ENCODING = "ISO-8859-1"
dataset = pd.read_csv(r'C:\Users\GEOMATIC\Desktop\mlflow\training.1600000.processed.noemoticon.csv', encoding=DATASET_ENCODING , names=DATASET_COLUMNS)

# Removing the unnecessary columns
dataset = dataset[['sentiment','text']]
# Replacing the values to ease understanding
dataset['sentiment'] = dataset['sentiment'].replace(4,1)

# Plotting the distribution for dataset
ax = dataset.groupby('sentiment').count().plot(kind='bar', title='Distribution of data', legend=False)
ax.set_xticklabels(['Negative','Positive'], rotation=0)

# Storing data in lists
text, sentiment = list(dataset['text']), list(dataset['sentiment'])

# Defining dictionary containing all emojis with their meanings.
emojis = {':)': 'smile', ':-)': 'smile', ';d': 'wink', ':-E': 'vampire', ':(': 'sad',
          ':-(': 'sad', ':-<': 'sad', ':P': 'raspberry', ':O': 'surprised',
          ':-@': 'shocked', ':@': 'shocked',':-$': 'confused', ':\\': 'annoyed',
          ':#': 'mute', ':X': 'mute', ':^)': 'smile', ':-&': 'confused', '$_$': 'greedy',
          '@@': 'eyeroll', ':-!': 'confused', ':-D': 'smile', ':-0': 'yell', 'O.o': 'confused',
          '<(-_-)>': 'robot', 'd[-_-]b': 'dj', ":'-)": 'sadsmile', ';)': 'wink',
          ';-)': 'wink', 'O:-)': 'angel','O*-)': 'angel','(:-D': 'gossip', '=^.^=': 'cat'}

## Defining set containing all stopwords in english.
stopwordlist = ['a', 'about', 'above', 'after', 'again', 'ain', 'all', 'am', 'an',
             'and','any','are', 'as', 'at', 'be', 'because', 'been', 'before',
             'being', 'below', 'between','both', 'by', 'can', 'd', 'did', 'do',
             'does', 'doing', 'down', 'during', 'each','few', 'for', 'from',
             'further', 'had', 'has', 'have', 'having', 'he', 'her', 'here',
             'hers', 'herself', 'him', 'himself', 'his', 'how', 'i', 'if', 'in',
             'into','is', 'it', 'its', 'itself', 'just', 'll', 'm', 'ma',
             'me', 'more', 'most','my', 'myself', 'now', 'o', 'of', 'on', 'once',
             'only', 'or', 'other', 'our', 'ours','ourselves', 'out', 'own', 're',
             's', 'same', 'she', "shes", 'should', "shouldve",'so', 'some', 'such',
             't', 'than', 'that', "thatll", 'the', 'their', 'theirs', 'them',
             'themselves', 'then', 'there', 'these', 'they', 'this', 'those',
             'through', 'to', 'too','under', 'until', 'up', 've', 'very', 'was',
             'we', 'were', 'what', 'when', 'where','which','while', 'who', 'whom',
             'why', 'will', 'with', 'won', 'y', 'you', "youd","youll", "youre",
             "youve", 'your', 'yours', 'yourself', 'yourselves']

# Processing data
def preprocess(textdata):
    print("Processing data...")
    processedText = []

    # Create Lemmatizer and Stemmer.
    wordLemm = WordNetLemmatizer()

    # Defining regex patterns.
    urlPattern        = r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)"
    userPattern       = '@[^\s]+'
    alphaPattern      = "[^a-zA-Z0-9]"
    sequencePattern   = r"(.)\1\1+"
    seqReplacePattern = r"\1\1"

    for tweet in textdata:
        tweet = tweet.lower()

        tweet = re.sub(urlPattern,' URL',tweet)
        for emoji in emojis.keys():
            tweet = tweet.replace(emoji, "EMOJI" + emojis[emoji])
        tweet = re.sub(userPattern,' USER', tweet)
        tweet = re.sub(alphaPattern, " ", tweet)
        tweet = re.sub(sequencePattern, seqReplacePattern, tweet)

        tweetwords = ''
        for word in tweet.split():
            if len(word)>1:
                # Lemmatizing the word.
                word = wordLemm.lemmatize(word)
                tweetwords += (word+' ')

        processedText.append(tweetwords)

    return processedText

# Start MLflow run
with mlflow.start_run():
    # Log dataset and preprocessing step
    t = time.time()
    processedtext = preprocess(text)
    mlflow.log_param("preprocessing_time_seconds", round(time.time() - t, 2))
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(processedtext, sentiment, test_size=0.05, random_state=0)
    vectoriser = TfidfVectorizer(ngram_range=(1, 2), max_features=500000)
    vectoriser.fit(X_train)
    
    X_train, X_test = vectoriser.transform(X_train), vectoriser.transform(X_test)
    mlflow.log_param("vectorizer_ngram_range", (1, 2))
    mlflow.log_param("vectorizer_max_features", 500000)

    # Define and train model
    LRmodel = LogisticRegression(C=2, max_iter=1000, n_jobs=-1)
    mlflow.log_param("model_type", "LogisticRegression")
    mlflow.log_param("C", 2)
    mlflow.log_param("max_iter", 1000)
    
    t = time.time()
    LRmodel.fit(X_train, y_train)
    training_time = round(time.time() - t, 2)
    mlflow.log_metric("training_time_seconds", training_time)

    # Model evaluation
    y_pred = LRmodel.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    mlflow.log_metric("accuracy", report["accuracy"])
    mlflow.log_metric("precision", report["weighted avg"]["precision"])
    mlflow.log_metric("recall", report["weighted avg"]["recall"])
    mlflow.log_metric("f1_score", report["weighted avg"]["f1-score"])

    # Confusion matrix plot
    cf_matrix = confusion_matrix(y_test, y_pred)
    sns.heatmap(cf_matrix, annot=True, cmap='Blues', fmt='', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig("confusion_matrix.png")
    plt.close()
    
    # Log model and confusion matrix
    mlflow.sklearn.log_model(LRmodel, "sentiment_analysis_model")
    mlflow.log_artifact("confusion_matrix.png")

    # Save model locally as pickle file
    with open("Sentiment-LR.pickle", "wb") as file:
        pickle.dump(LRmodel, file)
    mlflow.log_artifact("Sentiment-LR.pickle")

    print("Experiment logged successfully in MLflow!")