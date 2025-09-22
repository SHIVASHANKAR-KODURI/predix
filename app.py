from flask import Flask, request, jsonify
import joblib
import numpy as np
import sys
from flask_cors import CORS
from sklearn.datasets import load_breast_cancer
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Try to import Flask and other libraries
try:
    from flask import Flask, request, jsonify
    import joblib
    import numpy as np
    from flask_cors import CORS
    import pandas as pd
except ImportError:
    print("Error: Required libraries not found.")
    print("Please install them using pip:")
    print("pip install pandas scikit-learn flask numpy joblib flask-cors")
    sys.exit()

# Load the trained models
try:
    diabetes_model = joblib.load('diabetes_model.joblib')
    heartdisease_model = joblib.load('heartdisease_model.joblib')
    
    ckd_model_data = joblib.load('ckd_model.joblib')
    ckd_model = ckd_model_data['model']
    ckd_feature_names = ckd_model_data['feature_names']

    breastcancer_model = joblib.load('breastcancer_model.joblib')

    parkinsons_model_data = joblib.load('parkinsons_model.joblib')
    parkinsons_model = parkinsons_model_data['model']
    parkinsons_feature_names = parkinsons_model_data['feature_names']
    parkinsons_scaler = parkinsons_model_data['scaler']
    
    liver_model_data = joblib.load('liver_model.joblib')
    liver_model = liver_model_data['model']
    liver_feature_names = liver_model_data['feature_names']

except FileNotFoundError:
    print("Error: One or more model files not found.")
    print("Please run the corresponding Python scripts (e.g., diabetes_prediction.py) to train and save the models first.")
    sys.exit()

# Create the Flask application
app = Flask(__name__)
# Enable CORS to allow requests from your HTML file
CORS(app)

@app.route('/predict_diabetes', methods=['POST'])
def predict_diabetes():
    """
    API endpoint to predict diabetes based on user input and return feature importances.
    """
    try:
        data = request.json
        required_features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
        if not all(feature in data for feature in required_features):
            return jsonify({'error': 'Missing features'}), 400

        features = [
            data['Pregnancies'], data['Glucose'], data['BloodPressure'], data['SkinThickness'],
            data['Insulin'], data['BMI'], data['DiabetesPedigreeFunction'], data['Age']
        ]
        input_data = np.array([features])
        prediction = diabetes_model.predict(input_data)
        feature_importances = diabetes_model.feature_importances_.tolist()
        result = int(prediction[0])

        return jsonify({
            'prediction': result,
            'feature_importances': feature_importances,
            'feature_names': required_features
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_heartdisease', methods=['POST'])
def predict_heartdisease():
    """
    API endpoint to predict heart disease based on user input and return feature importances.
    """
    try:
        data = request.json
        required_features = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
        if not all(feature in data for feature in required_features):
            return jsonify({'error': 'Missing features'}), 400

        features = [
            data['age'], data['sex'], data['cp'], data['trestbps'], data['chol'], data['fbs'],
            data['restecg'], data['thalach'], data['exang'], data['oldpeak'], data['slope'], data['ca'], data['thal']
        ]
        input_data = np.array([features])
        prediction = heartdisease_model.predict(input_data)
        feature_importances = heartdisease_model.feature_importances_.tolist()
        result = int(prediction[0])

        return jsonify({
            'prediction': result,
            'feature_importances': feature_importances,
            'feature_names': required_features
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_ckd', methods=['POST'])
def predict_ckd():
    """
    API endpoint to predict chronic kidney disease based on user input and return feature importances.
    """
    try:
        data = request.json
        required_features_from_form = [
            'age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 'bu', 'sc', 'sod', 'pot',
            'hemo', 'pcv', 'wc', 'rc', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane'
        ]

        if not all(feature in data for feature in required_features_from_form):
            return jsonify({'error': 'Missing features'}), 400

        input_features_dict = {f: 0.0 for f in ckd_feature_names}
        for f in ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc']:
            input_features_dict[f] = data.get(f, 0.0)

        input_features_dict['rbc_normal'] = 1 if data.get('rbc') == 1 else 0
        input_features_dict['pc_normal'] = 1 if data.get('pc') == 1 else 0
        input_features_dict['pcc_present'] = 1 if data.get('pcc') == 1 else 0
        input_features_dict['ba_present'] = 1 if data.get('ba') == 1 else 0
        input_features_dict['htn_yes'] = 1 if data.get('htn') == 1 else 0
        input_features_dict['dm_yes'] = 1 if data.get('dm') == 1 else 0
        input_features_dict['cad_yes'] = 1 if data.get('cad') == 1 else 0
        input_features_dict['appet_good'] = 1 if data.get('appet') == 1 else 0
        input_features_dict['pe_yes'] = 1 if data.get('pe') == 1 else 0
        input_features_dict['ane_yes'] = 1 if data.get('ane') == 1 else 0
        
        features = [input_features_dict[f] for f in ckd_feature_names]
        input_data = np.array([features])
        
        prediction = ckd_model.predict(input_data)
        feature_importances = ckd_model.feature_importances_.tolist()
        result = int(prediction[0])

        return jsonify({
            'prediction': result,
            'feature_importances': feature_importances,
            'feature_names': ckd_feature_names
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_breastcancer', methods=['POST'])
def predict_breastcancer():
    """
    API endpoint to predict breast cancer based on user input and return feature importances.
    """
    try:
        data = request.json
        required_features = [
            'mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness', 
            'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry', 
            'mean fractal dimension', 'radius error', 'texture error', 'perimeter error',
            'area error', 'smoothness error', 'compactness error', 'concavity error',
            'concave points error', 'symmetry error', 'fractal dimension error',
            'worst radius', 'worst texture', 'worst perimeter', 'worst area',
            'worst smoothness', 'worst compactness', 'worst concavity',
            'worst concave points', 'worst symmetry', 'worst fractal dimension'
        ]
        
        if not all(feature in data for feature in required_features):
            return jsonify({'error': 'Missing features'}), 400

        features = [data[f] for f in required_features]
        input_data = np.array([features])
        
        prediction = breastcancer_model.predict(input_data)
        feature_importances = breastcancer_model.feature_importances_.tolist()
        result = int(prediction[0])

        return jsonify({
            'prediction': result,
            'feature_importances': feature_importances,
            'feature_names': required_features
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_parkinsons', methods=['POST'])
def predict_parkinsons():
    """
    API endpoint to predict Parkinson's disease based on user input.
    """
    try:
        data = request.json
        required_features = [
            'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)',
            'MDVP:Rap', 'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer', 'MDVP:Shimmer(dB)',
            'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR',
            'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE'
        ]
        
        if not all(feature in data for feature in required_features):
            return jsonify({'error': 'Missing features'}), 400

        features = [data[f] for f in required_features]
        
        scaled_features = parkinsons_scaler.transform(np.array(features).reshape(1, -1))
        
        prediction = parkinsons_model.predict(scaled_features)
        feature_importances = parkinsons_model.feature_importances_.tolist()
        result = int(prediction[0])

        return jsonify({
            'prediction': result,
            'feature_importances': feature_importances,
            'feature_names': required_features
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict_liver', methods=['POST'])
def predict_liver():
    """
    API endpoint to predict liver disease based on user input.
    """
    try:
        data = request.json
        required_features = [
            'Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin', 
            'Alkaline_Phosphotase', 'Alamine_Aminotransferase', 'Aspartate_Aminotransferase', 
            'Total_Protiens', 'Albumin', 'Albumin_and_Globulin_Ratio'
        ]
        
        if not all(feature in data for feature in required_features):
            return jsonify({'error': 'Missing features'}), 400

        features = [data[f] for f in required_features]
        input_data = np.array([features])
        
        prediction = liver_model.predict(input_data)
        feature_importances = liver_model.feature_importances_.tolist()
        result = int(prediction[0])

        return jsonify({
            'prediction': result,
            'feature_importances': feature_importances,
            'feature_names': required_features
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':

    app.run(host='0.0.0.0',debug=True)
