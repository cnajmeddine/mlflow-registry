import kagglehub
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import re
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from nltk.corpus import stopwords
from mlflow.models import infer_signature

# Set default MLflow experiment
mlflow.set_experiment("Sentiment Analysis Experiment")

# Download Sentiment140 dataset
print("Downloading Sentiment140 dataset...")
path = kagglehub.dataset_download("kazanova/sentiment140")
file_path = f"{path}/training.1600000.processed.noemoticon.csv"
print("Path to dataset files:", file_path)

# Download NLTK data files
print("Downloading NLTK data files...")
nltk.download("stopwords")
nltk.download("punkt")
nltk.download("punkt_tab")
print("Downloaded NLTK files")

# Load Sentiment140 dataset
def load_data(file_path):
    print("Loading data...")
    col_names = ["target", "ids", "date", "flag", "user", "text"]
    df = pd.read_csv(file_path, encoding="latin-1", names=col_names)
    print("Data loaded successfully.")
    return df

# Preprocess text data
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    tokens = nltk.word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words("english")]
    return " ".join(tokens)

# Prepare dataset
def prepare_data(df):
    print("Preprocessing and preparing data...")
    df["cleaned_text"] = df["text"].apply(preprocess_text)
    label_encoder = LabelEncoder()
    df["target"] = label_encoder.fit_transform(df["target"])
    X_train, X_test, y_train, y_test = train_test_split(df["cleaned_text"], df["target"], test_size=0.2, random_state=42)
    print("Data prepared successfully.")
    return X_train, X_test, y_train, y_test

# Feature extraction using TF-IDF
def vectorize_data(X_train, X_test):
    print("Vectorizing data with TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    print("Data vectorized successfully.")
    return X_train_vec, X_test_vec

# Train model and log to MLflow
def train_model(X_train, y_train, X_test, y_test, C=1.0):
    print("Training model...")
    model = LogisticRegression(C=C, max_iter=100)
    model.fit(X_train, y_train)
    print("Model trained successfully.")

    # Predictions and metrics
    print("Evaluating model...")
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average="weighted")
    recall = recall_score(y_test, predictions, average="weighted")
    f1 = f1_score(y_test, predictions, average="weighted")
    print(f"Model evaluation complete. Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1 Score: {f1}")

    # Log parameters, metrics, and model with MLflow
    print("Logging model and metrics to MLflow...")
    with mlflow.start_run() as run:
        # Log hyperparameters
        mlflow.log_param("C", C)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.set_tag("Training Info", "Sentiment Analysis with Logistic Regression")
        
        # Infer model signature
        signature = infer_signature(X_train, model.predict(X_train))
        
        # Log the model
        model_info = mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="sentiment_model",
            signature=signature,
            input_example=X_train[:5],  # Example input for logging
            registered_model_name="SentimentAnalysisModel"
        )
        
    print(f"Run ID: {run.info.run_id}")
    print("Model and metrics logged to MLflow.")

# Main function to execute pipeline
def main(file_path):
    print("Starting main pipeline...")
    df = load_data(file_path)
    X_train, X_test, y_train, y_test = prepare_data(df)
    X_train_vec, X_test_vec = vectorize_data(X_train, X_test)
    print("Data split")
    train_model(X_train_vec, y_train, X_test_vec, y_test)
    print("Pipeline execution completed.")

# Run pipeline
main(file_path)
