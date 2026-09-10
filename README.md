# Predix

Predix is a responsive machine-learning health risk prediction web application built with **Python, Flask, scikit-learn, HTML, CSS and JavaScript**. It provides six model-backed prediction modules with local form memory and visual feature-importance summaries.

## Included models

- Diabetes
- Heart Disease
- Chronic Kidney Disease
- Breast Cancer
- Parkinson's Disease
- Liver Disease

## Run locally

```bash
pip install -r requirements.txt
python app.py
```

Open **http://127.0.0.1:5000**.

## Demo data

Every prediction page has a **Use demo data** button. It fills the complete form with a valid example dataset. Each applicable field also shows an example value beside its label.

## Deployment

The project is deployment-ready for platforms that run Docker or Gunicorn. `Dockerfile` is included for Railway, Koyeb, Fly.io and similar services. The app reads the platform `PORT` automatically.

For a fast portfolio/interview deployment, use an always-on paid small instance rather than a sleeping free instance. Railway Hobby is the simplest option; Koyeb is a good lower-cost alternative.

## Performance

All six trained models are loaded once when the server starts. Prediction requests do not load model files repeatedly. The frontend uses only local HTML/CSS/JavaScript and the result feature-importance chart is rendered without external chart libraries, so it works reliably after deployment.

## Important

Predix is an educational/portfolio project. Its outputs are model predictions, not medical diagnoses or treatment advice. Do not use the results to make medical decisions.
