
from fastapi import FastAPI
import pickle
import numpy as np

app = FastAPI()
model = pickle.load(open("models/model.pkl", "rb"))

@app.post("/predict")
def predict(features: dict):
    data = np.array([list(features.values())])
    pred = model.predict(data)[0]
    return {"prediction": int(pred)}
