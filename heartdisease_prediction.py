import pandas as pd
import sys
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

def load_data():
    """
    Loads and preprocesses the Heart Disease Dataset from a CSV file.
    Assumes the file 'heart.csv' is in the same directory.
    """
    try:
        df = pd.read_csv('heart.csv')
        return df
    except FileNotFoundError:
        print("Error: 'heart.csv' not found.")
        print("Please download the file from https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset and place it in the same directory.")
        return None

# Load the dataset
df = load_data()

if df is not None:
    # Separate features and target variable.
    # The 'target' column is what we want to predict.
    X = df.drop('target', axis=1)
    y = df['target']

    # Split data into training and testing sets.
    # A test size of 0.2 means 20% of the data is used for testing.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the Random Forest Classifier.
    # This model is a good choice for this type of problem and provides feature importances.
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model on the test data.
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f'Model Accuracy: {accuracy:.2f}')

    # Save the trained model to a file so it can be used by the Flask application.
    joblib.dump(model, 'heartdisease_model.joblib')
    print("Model saved as 'heartdisease_model.joblib'")
else:
    print("Model training aborted due to missing dataset file.")