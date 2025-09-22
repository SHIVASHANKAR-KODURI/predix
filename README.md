### **Predix**

This is a web-based system that uses machine learning models to predict the likelihood of various diseases based on user-provided health data. The project features a clean, responsive user interface and a Python-Flask backend that serves the prediction models.

-----

### **Table of Contents**

  * [Features](https://www.google.com/search?q=%23features)
  * [Technologies Used](https://www.google.com/search?q=%23technologies-used)
  * [Project Structure](https://www.google.com/search?q=%23project-structure)
  * [How to Run the Project](https://www.google.com/search?q=%23how-to-run-the-project)
      * [Prerequisites](https://www.google.com/search?q=%23prerequisites)
      * [Step 1: Set up the Backend](https://www.google.com/search?q=%23step-1-set-up-the-backend)
      * [Step 2: Run the Frontend](https://www.google.com/search?q=%23step-2-run-the-frontend)
  * [API Endpoints](https://www.google.com/search?q=%23api-endpoints)
  * [Deployment](https://www.google.com/search?q=%23deployment)
  * [Contributing](https://www.google.com/search?q=%23contributing)

-----

### **Features**

  * **Multi-Disease Prediction**: Predicts the likelihood of several diseases including Diabetes, Heart Disease, Chronic Kidney Disease, Breast Cancer, and Parkinson's Disease.
  * **User-Friendly Interface**: A clean, responsive design with a glassmorphism effect for an intuitive user experience.
  * **Visual Outputs**: Displays a bar chart of feature importance for each prediction, helping users understand which factors most influenced the model's result.
  * **Modular Backend**: A scalable Flask API that serves multiple machine learning models from a single server.

-----

### **Technologies Used**

#### **Frontend**

  * **HTML**: For the website structure.
  * **CSS (Tailwind CSS)**: For styling and responsive design.
  * **JavaScript**: For handling user input, making API calls, and rendering dynamic content.
  * **Chart.js**: A powerful JavaScript library for creating interactive charts and graphs.

#### **Backend**

  * **Python**: The core language for the backend logic and machine learning models.
  * **Flask**: A lightweight web framework for creating the prediction API.
  * **Scikit-learn**: The machine learning library used for training the prediction models.
  * **Pandas**: For data manipulation and preprocessing.
  * **Joblib**: For saving and loading the trained machine learning models.
  * **Flask-CORS**: To handle Cross-Origin Resource Sharing, allowing the frontend and backend to communicate.

-----

### **Project Structure**

The repository contains the following key files:

  * `index.html`: The main landing page with links to each disease prediction form.
  * `diabetes_form.html`: The frontend form for diabetes prediction.
  * `heartdisease_form.html`: The frontend form for heart disease prediction.
  * `ckd_form.html`: The frontend form for chronic kidney disease prediction.
  * `breastcancer_form.html`: The frontend form for breast cancer prediction.
  * `parkinsons_form.html`: The frontend form for Parkinson's disease prediction.
  * `liver_form.html`: The frontend form for liver disease prediction.
  * `app.py`: The Flask backend application that runs the prediction API.
  * `[disease]_prediction.py`: Separate scripts (e.g., `diabetes_prediction.py`) used to train and save the machine learning models.
  * `[disease]_model.joblib`: The saved machine learning model files (these are generated after running the prediction scripts).

-----

### **How to Run the Project**

#### **Prerequisites**

  * Python 3.x
  * pip (Python package installer)

Clone the repository to your local machine:

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

#### **Step 1: Set up the Backend**

1.  **Install Python Libraries**: Install all the necessary packages for the backend.

    ```bash
    pip install Flask scikit-learn pandas joblib Flask-CORS numpy
    ```

2.  **Download Datasets**: Download the following datasets and place them in your project directory:

      * **Diabetes**: [Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/mathchi/diabetes-data-set) (`diabetes.csv`)
      * **Heart Disease**: [Heart Disease Dataset](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset) (`heart.csv`)
      * **CKD**: [CKD Dataset](https://www.kaggle.com/datasets/mansoordaku/ckdisease) (`kidney_disease.csv`)
      * **Parkinson's**: [Parkinson's Dataset](https://www.kaggle.com/datasets/vikasukani/parkinsons-disease-data-set) (`parkinsons.data`)
      * **Liver Disease**: [Liver Patient Dataset](https://www.kaggle.com/datasets/uciml/indian-liver-patient-records) (`indian_liver_patient.csv`)

3.  **Train the Models**: Run each of the prediction scripts to generate the `.joblib` model files.

    ```bash
    python diabetes_prediction.py
    python heartdisease_prediction.py
    python ckd_prediction.py
    python breastcancer_prediction.py
    python parkinsons_prediction.py
    python liver_prediction.py
    ```

4.  **Start the Flask Server**: Once all the `.joblib` files are created, start the Flask application.

    ```bash
    python app.py
    ```

    The server will run on `http://127.0.0.1:5000`.

#### **Step 2: Run the Frontend**

Simply open the `index.html` file in your web browser. Make sure the Flask server is running in the background.

-----

### **API Endpoints**

The Flask application exposes the following endpoints:

  * `POST /predict_diabetes`: Predicts diabetes.
  * `POST /predict_heartdisease`: Predicts heart disease.
  * `POST /predict_ckd`: Predicts chronic kidney disease.
  * `POST /predict_breastcancer`: Predicts breast cancer.
  * `POST /predict_parkinsons`: Predicts Parkinson's disease.
  * `POST /predict_liver`: Predicts liver disease.

-----

### **Deployment**

The frontend can be deployed to a static hosting service like Netlify. The backend, since it requires a Python server, can be deployed to platforms like Heroku, Render, or Google Cloud Functions.

-----

### **Contributing**

Feel free to fork the repository, make improvements, and submit a pull request\!
