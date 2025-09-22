import pandas as pd
import sys
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

def load_data():
    """
    Loads and preprocesses the Indian Liver Patient Dataset.
    Assumes the file 'indian_liver_patient.csv' is in the same directory.
    """
    try:
        df = pd.read_csv('indian_liver_patient.csv')
        
        # The dataset already has headers, so no need to rename.
        # However, we will check if the expected columns exist.
        expected_cols = [
            'Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin', 
            'Alkaline_Phosphotase', 'Alamine_Aminotransferase', 'Aspartate_Aminotransferase', 
            'Total_Protiens', 'Albumin', 'Albumin_and_Globulin_Ratio', 'Dataset'
        ]
        if not all(col in df.columns for col in expected_cols):
            print("Error: The CSV file does not contain the expected column names.")
            return None, None, None

        # Map gender column (Male=1, Female=0).
        df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})

        # The target column is named 'Dataset' and needs to be mapped from 1 and 2 to 1 and 0.
        df['Dataset'] = df['Dataset'].map({1: 1, 2: 0})
        
        # Fill missing values with the mean of the column.
        df = df.fillna(df.mean(numeric_only=True))
        
        # Define features and target.
        X = df.drop('Dataset', axis=1)
        y = df['Dataset']
        
        return X, y, X.columns.tolist()
        
    except FileNotFoundError:
        print("Error: 'indian_liver_patient.csv' not found.")
        print("Please download the file from https://www.kaggle.com/datasets/uciml/indian-liver-patient-records and place it in the same directory.")
        return None, None, None
    except Exception as e:
        print(f"Error processing the dataset: {e}")
        return None, None, None

# Load the dataset
X, y, feature_names = load_data()

if X is not None:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f'Model Accuracy: {accuracy:.2f}')
    
    joblib.dump({'model': model, 'feature_names': feature_names}, 'liver_model.joblib')
    print("Model and feature names saved as 'liver_model.joblib'")
else:
    print("Model training aborted due to a dataset issue.")