import pandas as pd
import sys
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

def load_data():
    """
    Loads and preprocesses the Chronic Kidney Disease (CKD) dataset from a CSV file.
    Assumes the file 'kidney_disease.csv' is in the same directory.
    """
    try:
        df = pd.read_csv('kidney_disease.csv')
        
        # Handle missing values by replacing '?' and '\t' with NaN
        df = df.replace(['?', '\t'], pd.NA)

        # Drop the 'id' column as it's not needed for prediction
        df = df.drop('id', axis=1)

        # Separate categorical and numerical columns for processing
        categorical_cols = ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']
        numerical_cols = [col for col in df.columns if col not in categorical_cols + ['classification']]

        # Convert numerical columns to numeric type
        for col in numerical_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # One-hot encode categorical columns and drop first to avoid multicollinearity
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

        # Fill remaining missing numerical values with the mean of the column
        df = df.fillna(df.mean(numeric_only=True))

        # Rename target column and map string values to numbers
        df = df.rename(columns={'classification': 'target'})
        df['target'] = df['target'].map({'ckd\t': 1, 'notckd': 0, 'ckd': 1})
        df = df.dropna(subset=['target'])

        return df
    except FileNotFoundError:
        print("Error: 'kidney_disease.csv' not found.")
        print("Please download the dataset from a source like Kaggle and place it in the same directory.")
        return None

# Load the dataset
df = load_data()

if df is not None:
    # Separate features and target variable
    X = df.drop('target', axis=1)
    y = df['target']
    
    # We'll save the list of features so that our Flask app knows the exact order
    feature_names = X.columns.tolist()

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the Random Forest Classifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f'Model Accuracy: {accuracy:.2f}')

    # Save the trained model and the feature names to a file
    joblib.dump({'model': model, 'feature_names': feature_names}, 'ckd_model.joblib')
    print("Model and feature names saved as 'ckd_model.joblib'")
else:
    print("Model training aborted due to missing dataset file.")
