---
title: Heart Disease Detection
emoji: ❤️
colorFrom: red
colorTo: pink
sdk: docker
app_port: 8501
pinned: false
---

# ❤️ Heart Disease Detection

A Streamlit web app that predicts the likelihood of heart disease from clinical
data using a tuned Random Forest classifier.

## Features
- Patient input form with validation
- Prediction with disease / no-disease probability
- Responsive dashboard UI
- Pre-trained model (no retraining in the app)

## Project structure
```
Model/            RF_heart.pkl, scaler.pkl, feature_columns.pkl
app.py            Streamlit application
check_env.py      Version / file checker
requirements.txt
```

## Run locally
```bash
python -m venv venv
venv\Scripts\activate        # Windows  (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt
streamlit run app.py
```

## Run with Docker
```bash
docker build -t heart-app .
docker run -p 8501:8501 heart-app
```

## Deploy
Streamlit Community Cloud: push to GitHub, then at share.streamlit.io choose
the repo, branch `main`, main file `app.py`.

## Disclaimer
For educational and research purposes only. Not a medical device and not a
substitute for professional medical diagnosis or advice.