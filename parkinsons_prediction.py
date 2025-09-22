import pandas as pd
import sys
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
import joblib

def load_data():
    """
    Loads and preprocesses the Parkinson's dataset.
    Assumes the file 'parkinsons.data' is in the same directory.
    """
    try:
        df = pd.read_csv('parkinsons.data')
        
        # Drop the 'name' column as it is not needed for prediction.
        df = df.drop('name', axis=1)

        # Separate features (X) and target (y).
        # The 'status' column is the target (1=Parkinson's, 0=Healthy).
        X = df.drop('status', axis=1)
        y = df['status']

        # Scale the features to a range of [-1, 1].
        # This is a common and important step for this dataset.
        scaler = MinMaxScaler(feature_range=(-1, 1))
        X_scaled = scaler.fit_transform(X)
        
        # Return the scaled features, target variable, feature names, and the scaler
        # so it can be used for new data predictions in the Flask app.
        return X, y, X.columns.tolist(), scaler
        
    except FileNotFoundError:
        print("Error: 'parkinsons.data' not found.")
        print("Please download the file from https://www.kaggle.com/datasets/vikasukani/parkinsons-disease-data-set and place it in the same directory.")
        return None, None, None, None
    except Exception as e:
        print(f"Error processing the dataset: {e}")
        return None, None, None, None

# Load the dataset
X, y, feature_names, scaler = load_data()

if X is not None:
    # Split data into training and testing sets.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the Random Forest Classifier.
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model.
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f'Model Accuracy: {accuracy:.2f}')

    # Save the trained model, feature names, and the scaler to a file.
    # The scaler is important for preprocessing new user data in the Flask app.
    joblib.dump({'model': model, 'feature_names': feature_names, 'scaler': scaler}, 'parkinsons_model.joblib')
    print("Model, feature names, and scaler saved as 'parkinsons_model.joblib'")
else:
    print("Model training aborted due to a dataset issue.")