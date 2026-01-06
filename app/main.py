
from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI(title="Heart Disease Prediction API")
model = joblib.load("models/model.pkl")

@app.post("/predict")
def predict(features: dict):
    data = np.array([list(features.values())])
    pred = model.predict(data)[0]
    prob = model.predict_proba(data)[0][1]
    return {"prediction": int(pred), "confidence": float(prob)}
