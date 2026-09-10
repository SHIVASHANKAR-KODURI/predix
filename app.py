from flask import Flask, request, jsonify, send_from_directory
import os
import warnings
import joblib
import numpy as np
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but RandomForestClassifier was fitted with feature names",
    category=UserWarning,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, static_folder=None)
@app.after_request
def add_cache_headers(response):
    # Static UI assets are immutable within a deployment and can be cached by
    # the browser, reducing repeat-load latency without caching API responses.
    if request.path.endswith((".html", ".css", ".js", ".png", ".jpg", ".jpeg", ".svg", ".ico")):
        response.headers["Cache-Control"] = "public, max-age=300, stale-while-revalidate=86400"
    return response

# Models are loaded once when the Flask worker starts.
# This is intentionally eager: prediction itself should be fast and never pay
# the joblib loading cost on the user's first click.
_MODEL_CACHE = {}
_MODEL_READY = False

MODEL_FILES = {
    "diabetes": "diabetes_model.joblib",
    "heartdisease": "heartdisease_model.joblib",
    "ckd": "ckd_model.joblib",
    "breastcancer": "breastcancer_model.joblib",
    "parkinsons": "parkinsons_model.joblib",
    "liver": "liver_model.joblib",
}

def _load_model(name):
    if name in _MODEL_CACHE:
        return _MODEL_CACHE[name]
    path = os.path.join(BASE_DIR, MODEL_FILES[name])
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {MODEL_FILES[name]}")
    model = joblib.load(path)
    _MODEL_CACHE[name] = model
    return model

def _warm_models():
    global _MODEL_READY
    for name in MODEL_FILES:
        _load_model(name)
    _MODEL_READY = True

try:
    _warm_models()
except Exception:
    app_ready_error = True
    # Keep the process alive so the health endpoint can expose the failure.
    # Individual prediction endpoints will return a useful 500 instead of
    # silently hiding the startup problem.
else:
    app_ready_error = False

def _importance(model, fallback_names):
    values = getattr(model, "feature_importances_", None)
    if values is None:
        return [], fallback_names
    return np.asarray(values, dtype=float).tolist(), fallback_names


def _payload(data, required):
    if not isinstance(data, dict):
        return None, "Request body must be JSON."
    missing = [f for f in required if f not in data]
    if missing:
        return None, "Missing features: " + ", ".join(missing[:5])
    try:
        return [float(data[f]) for f in required], None
    except (TypeError, ValueError):
        return None, "All fields must contain valid numeric values."


@app.get("/")
def home():
    return send_from_directory(BASE_DIR, "index.html")


@app.get("/<path:page>")
def pages(page):
    # Serve only the project's HTML/CSS/JS/assets; API routes are defined above.
    if page.endswith((".html", ".css", ".js", ".png", ".jpg", ".jpeg", ".ico", ".svg")):
        full = os.path.join(BASE_DIR, page)
        if os.path.isfile(full):
            return send_from_directory(BASE_DIR, page)
    return jsonify({"error": "Page not found"}), 404


@app.post("/predict_diabetes")
def predict_diabetes():
    required = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    features, error = _payload(request.get_json(silent=True), required)
    if error:
        return jsonify({"error": error}), 400
    try:
        model = _load_model("diabetes")
        result = int(model.predict(np.array([features]))[0])
        imp, names = _importance(model, required)
        return jsonify({"prediction": result, "feature_importances": imp, "feature_names": names})
    except Exception as e:
        app.logger.exception("Diabetes prediction failed")
        return jsonify({"error": str(e)}), 500


@app.post("/predict_heartdisease")
def predict_heartdisease():
    required = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    features, error = _payload(request.get_json(silent=True), required)
    if error:
        return jsonify({"error": error}), 400
    try:
        model = _load_model("heartdisease")
        result = int(model.predict(np.array([features]))[0])
        imp, names = _importance(model, required)
        return jsonify({"prediction": result, "feature_importances": imp, "feature_names": names})
    except Exception as e:
        app.logger.exception("Heart prediction failed")
        return jsonify({"error": str(e)}), 500


@app.post("/predict_ckd")
def predict_ckd():
    required = ['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Request body must be JSON."}), 400
    missing = [f for f in required if f not in data]
    if missing:
        return jsonify({"error": "Missing features: " + ", ".join(missing[:5])}), 400
    try:
        model_data = _load_model("ckd")
        model = model_data['model']
        feature_names = model_data['feature_names']
        mapped = {f: 0.0 for f in feature_names}
        for f in ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc']:
            mapped[f] = float(data.get(f, 0.0))
        mapped['rbc_normal'] = 1 if int(data['rbc']) == 1 else 0
        mapped['pc_normal'] = 1 if int(data['pc']) == 1 else 0
        mapped['pcc_present'] = 1 if int(data['pcc']) == 1 else 0
        mapped['ba_present'] = 1 if int(data['ba']) == 1 else 0
        mapped['htn_yes'] = 1 if int(data['htn']) == 1 else 0
        mapped['dm_yes'] = 1 if int(data['dm']) == 1 else 0
        mapped['cad_yes'] = 1 if int(data['cad']) == 1 else 0
        mapped['appet_good'] = 1 if int(data['appet']) == 1 else 0
        mapped['pe_yes'] = 1 if int(data['pe']) == 1 else 0
        mapped['ane_yes'] = 1 if int(data['ane']) == 1 else 0
        features = [mapped[f] for f in feature_names]
        result = int(model.predict(np.array([features]))[0])
        imp, names = _importance(model, feature_names)
        return jsonify({"prediction": result, "feature_importances": imp, "feature_names": names})
    except Exception as e:
        app.logger.exception("CKD prediction failed")
        return jsonify({"error": str(e)}), 500


@app.post("/predict_breastcancer")
def predict_breastcancer():
    required = ['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness', 'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry', 'mean fractal dimension', 'radius error', 'texture error', 'perimeter error', 'area error', 'smoothness error', 'compactness error', 'concavity error', 'concave points error', 'symmetry error', 'fractal dimension error', 'worst radius', 'worst texture', 'worst perimeter', 'worst area', 'worst smoothness', 'worst compactness', 'worst concavity', 'worst concave points', 'worst symmetry', 'worst fractal dimension']
    features, error = _payload(request.get_json(silent=True), required)
    if error:
        return jsonify({"error": error}), 400
    try:
        model = _load_model("breastcancer")
        result = int(model.predict(np.array([features]))[0])
        imp, names = _importance(model, required)
        return jsonify({"prediction": result, "feature_importances": imp, "feature_names": names})
    except Exception as e:
        app.logger.exception("Breast cancer prediction failed")
        return jsonify({"error": str(e)}), 500


@app.post("/predict_parkinsons")
def predict_parkinsons():
    required = ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)', 'MDVP:Jitter(Abs)', 'MDVP:Rap', 'MDVP:PPQ', 'Jitter:DDP', 'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5', 'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE']
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "Request body must be JSON."}), 400
    missing = [f for f in required if f not in data]
    if missing:
        return jsonify({"error": "Missing features: " + ", ".join(missing[:5])}), 400
    try:
        model_data = _load_model("parkinsons")
        model = model_data['model']
        scaler = model_data['scaler']
        features = [float(data[f]) for f in required]
        scaled = scaler.transform(np.array(features).reshape(1, -1))
        result = int(model.predict(scaled)[0])
        imp, names = _importance(model, required)
        return jsonify({"prediction": result, "feature_importances": imp, "feature_names": names})
    except Exception as e:
        app.logger.exception("Parkinson's prediction failed")
        return jsonify({"error": str(e)}), 500


@app.post("/predict_liver")
def predict_liver():
    required = ['Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin', 'Alkaline_Phosphotase', 'Alamine_Aminotransferase', 'Aspartate_Aminotransferase', 'Total_Protiens', 'Albumin', 'Albumin_and_Globulin_Ratio']
    features, error = _payload(request.get_json(silent=True), required)
    if error:
        return jsonify({"error": error}), 400
    try:
        model_data = _load_model("liver")
        model = model_data['model']
        result = int(model.predict(np.array([features]))[0])
        imp, names = _importance(model, required)
        return jsonify({"prediction": result, "feature_importances": imp, "feature_names": names})
    except Exception as e:
        app.logger.exception("Liver prediction failed")
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
