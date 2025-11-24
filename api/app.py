from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
import os

app = FastAPI(title="Breast Cancer Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🔥 FIXED - always resolves correctly on Render
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "breast_cancer_best_pipeline.joblib")
model = joblib.load(MODEL_PATH)


class CancerInput(BaseModel):
    data: list


@app.get("/")
def home():
    return {"message": "Breast Cancer API is running"}


@app.post("/predict")
def predict(input_data: CancerInput):
    arr = np.array(input_data.data).reshape(1, -1)
    pred = model.predict(arr)[0]
    cancer = True if pred == 0 else False
    return {"cancer": cancer}
