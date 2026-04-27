# Steel Plate Defect — Root Cause Analysis

A Streamlit web app that classifies steel plate surface defects (7 types) using the UCI Steel Plates Faults dataset and performs root cause analysis via feature importance.

## Features
- **3 Models**: Logistic Regression, Random Forest, XGBoost (all trained with SMOTE for class balancing)
- **Model Comparison** table with Accuracy & Macro F1
- **Per-class metrics** and **Confusion Matrix** for any selected model
- **Actual vs Predicted** count plots across all three models
- **Root Cause Analysis** — top feature importance chart (RF & XGBoost)
- **Live Prediction** — enter custom feature values and get a real-time prediction with probability breakdown

## Deploy on Streamlit Community Cloud

1. Fork / push this repo to your GitHub account.
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Select your repo, branch `main`, and set **Main file path** to `app.py`.
4. Click **Deploy** — done!

> **Note:** The first load trains all three models (~1–2 min). Results are cached, so subsequent visits are instant.

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```
