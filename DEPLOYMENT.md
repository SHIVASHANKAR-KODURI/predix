# Predix deployment guide

## Best choice for fast interview/demo predictions

### 1. Railway — recommended

Use Railway if you want the simplest deployment and an always-running small service. The current project includes `Dockerfile` and `railway.toml`. Connect the GitHub repository, create a service from the repo, and deploy. The container automatically uses Railway's `PORT`.

For consistently fast responses, use the paid Hobby plan rather than relying on a sleeping/free service.

### 2. Koyeb — good lower-cost alternative

Koyeb supports Flask/Docker and has a free instance, but its free instance has only 0.1 vCPU and scales to zero after an hour without traffic. For fast ML predictions, use an always-on paid Eco instance instead.

### 3. Fly.io — fast and flexible

Fly.io runs containerized apps close to users and bills by usage. It is more hands-on than Railway. It is a good choice if you are comfortable with Docker and want regional control.

## Why this build is faster

- All six `.joblib` models are loaded once when the server starts.
- A prediction request does not load a model from disk.
- The frontend and API are served by the same Flask process, so there is no browser-to-third-party API hop.
- The feature-importance chart uses plain HTML/CSS/JavaScript, with no chart CDN or external JavaScript dependency.
- The Docker image runs Gunicorn with one worker and four threads, which is suitable for this small model-serving demo.

## Local test

```bash
pip install -r requirements.txt
python app.py
```

Then open `http://127.0.0.1:5000`.

## Important

Predix is an educational/portfolio project. Model results are informational predictions, not medical diagnoses.
