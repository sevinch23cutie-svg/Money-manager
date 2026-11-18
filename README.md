# AI Money Manager

A Streamlit web application demonstrating expense categorization using TF-IDF + Logistic Regression and unusual-amount detection using Isolation Forest.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy

Upload `app.py` and `requirements.txt` to a public GitHub repository. At https://share.streamlit.io create an app using the repository and `app.py` as the entry point.

## Test

Try `Starbucks coffee` / 5500, `Bus fare` / 1600, and `University textbook` / 35000. Check the predicted category and edit it if incorrect. Add a very large expense and inspect the anomaly section. The included sample transactions are fictional.

## Limitations

The classification model is trained on a tiny hand-written English sample and will not generalize reliably. The anomaly detector flags statistical outliers and does not judge necessity. Data persists only for the current Streamlit session. No currency conversion or bank integrations are provided.
