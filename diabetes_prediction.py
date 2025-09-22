import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

def load_data():
    try:
        df = pd.read_csv('diabetes.csv')
        return df
    except FileNotFoundError:
        print("Error: 'diabetes.csv' not found.")
        print("Please download the file from https://www.kaggle.com/datasets/mathchi/diabetes-data-set and place it in the same directory.")
        return None

# Load the dataset
df = load_data()

if df is not None:
    # Separate features and target variable
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the Random Forest Classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f'Model Accuracy: {accuracy:.2f}')

    # Save the trained model to a file
    joblib.dump(model, 'diabetes_model.joblib')
    print("Model saved as 'diabetes_model.joblib'")
else:
    print("Model training aborted due to missing dataset file.")
