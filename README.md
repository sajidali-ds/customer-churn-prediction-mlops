# Customer Churn Prediction — ANN + ML Comparison + MLOps

A bank-customer churn classifier built with Keras/TensorFlow, benchmarked
against 7 classical ML algorithms, served through Streamlit, and wrapped
with a full MLOps layer: experiment tracking, model versioning, CI/CD, and
production monitoring with drift detection.

## What's included

- **No duplicated preprocessing logic** — training and inference share one
  `transform_features()` function.
- **Proper train/val/test split**, scaler/encoders fit on training data only.
- **Class weighting** for the ~80/20 churn imbalance.
- **Fair ML algorithm comparison** — ANN vs. Logistic Regression, Decision
  Tree, Random Forest, Gradient Boosting, KNN, Naive Bayes, SVM, XGBoost,
  all on the identical test split.
- **Experiment tracking (MLflow)** — every training run logs hyperparameters
  and test metrics, queryable and comparable later.
- **Model versioning (MLflow Model Registry)** — every trained model gets a
  version number (`churn-ann` v1, v2, ...) instead of silently overwriting
  a single file.
- **CI/CD (GitHub Actions)** — tests run automatically on every push; a
  Docker image builds and publishes automatically on merge to `main`.
- **Monitoring & drift detection** — every prediction the app makes is
  logged, and a statistical test (Kolmogorov–Smirnov) compares incoming
  data against the training distribution to flag when the model might need
  retraining.

## Project structure

```
churn_project/
├── .github/workflows/
│   ├── ci.yml                # runs pytest on every push/PR
│   └── cd.yml                # builds + pushes Docker image to GHCR on main
├── config.yaml                # single source of truth for paths & hyperparameters
├── requirements.txt
├── Churn_Modelling.csv        # dataset (already included)
├── src/
│   ├── utils.py               # config loading, seeding, json helpers
│   ├── data_preprocessing.py  # load/split/encode/scale, shared train+inference
│   ├── model.py                # ANN architecture
│   ├── train.py                # training pipeline + MLflow logging + model registry
│   ├── evaluate.py             # shared metric computation
│   ├── compare_models.py       # trains 7 classical ML models + XGBoost, compares vs ANN
│   ├── predict.py              # ChurnPredictor — inference + automatic prediction logging
│   └── monitoring.py           # prediction logging + KS-test drift detection
├── tests/
│   └── test_preprocessing.py
├── app.py                      # Streamlit UI: Predict / Model Comparison / Monitoring
├── artifacts/                   # generated: model, encoders, metrics, comparison, prediction log
├── mlflow.db                    # generated: MLflow tracking database (SQLite)
├── Dockerfile
└── README.md
```

## Sequence: what to run, in order

1. **Install dependencies.**
   ```bash
   pip install -r requirements.txt
   ```
2. **Train the ANN** (now also logs to MLflow and registers the model version).
   ```bash
   python -m src.train
   ```
   Produces the usual `artifacts/` files, plus `mlflow.db` (experiment
   tracking database) and registers a new version of `churn-ann` in the
   MLflow Model Registry.
3. **(New) View experiment tracking:**
   ```bash
   mlflow ui --backend-store-uri sqlite:///mlflow.db
   ```
   Open `http://localhost:5000` — see every training run's hyperparameters
   and metrics side by side, and the `churn-ann` model's version history
   under the "Models" tab. Leave this running in its own terminal.
4. **Run the model comparison** (must come after step 2):
   ```bash
   python -m src.compare_models
   ```
5. **(Optional) Run tests:**
   ```bash
   pytest tests/ -v
   ```
6. **Launch the app:**
   ```bash
   streamlit run app.py
   ```
   Sidebar now has **three** pages: **Predict**, **Model Comparison**, and
   **Monitoring**. Every prediction you make on the Predict page is
   automatically logged for the Monitoring page to analyze.
7. **(New) Check for drift** — on the Monitoring page, click "Run drift
   check" after making a handful of predictions. It compares your input
   distribution against training data and flags any feature that looks
   statistically different (a sign the model may need retraining on
   fresher data).
8. **(Optional) Docker, same as before:**
   ```bash
   docker build -t churn-app .
   docker run -p 8501:8501 churn-app
   ```

## Setting up CI/CD (GitHub Actions)

This only activates once the project is pushed to a GitHub repo — nothing
runs locally.

1. Push this project to your GitHub repo (replace the old repo contents,
   or create a new one).
2. That's it — `.github/workflows/ci.yml` runs automatically on every push
   and every pull request: installs dependencies, runs `pytest`, and
   compile-checks all source files. Check the "Actions" tab on your repo
   to watch it run.
3. `.github/workflows/cd.yml` runs on every push to `main`: builds the
   Docker image and publishes it to GitHub Container Registry
   (`ghcr.io/<your-username>/churn-app`) — no extra setup needed, it uses
   GitHub's automatically-provided token.
4. If you want a badge on your README showing "tests passing", GitHub
   gives you the markdown snippet under Actions → your workflow → "Create
   status badge".

## What you need to add yourself

- **Nothing to get it running locally** — dataset and trained artifacts
  ship with this package.
- **To activate CI/CD:** just push to a GitHub repository (step above) —
  no code changes needed.
- **To view experiment tracking:** run the `mlflow ui` command in step 3 —
  nothing to configure.
- **If you retrain repeatedly:** each run creates a new MLflow run and a
  new registered model version automatically — you don't need to do
  anything manually to "version" a model.
- **If you want real drift signal (not just the demo above):** the
  Monitoring page needs a reasonable number of real predictions logged
  (10+) before the KS test is meaningful — a couple of test clicks will
  show `drift_detected: true` simply because the sample is tiny and
  repetitive, which is expected, not a bug.

## Retraining with new data / hyperparameters

Edit `config.yaml` and re-run `python -m src.train` then
`python -m src.compare_models`. Every run is tracked separately in MLflow,
so you can compare old vs. new hyperparameters side by side in the MLflow
UI rather than only remembering the latest one.
