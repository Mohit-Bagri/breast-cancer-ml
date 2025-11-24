from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Breast Cancer Prediction API")

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all domains (Github Pages)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model (Render runs the app from project root)
model = joblib.load("../model/breast_cancer_best_pipeline.joblib")


class CancerInput(BaseModel):
    data: list


@app.get("/")
def home():
    return {"message": "Breast Cancer API is running"}


@app.post("/predict")
def predict(input_data: CancerInput):
    arr = np.array(input_data.data).reshape(1, -1)
    pred = model.predict(arr)[0]

    # 0 = malignant (cancer), 1 = benign (no cancer)
    cancer = True if pred == 0 else False

    return {"cancer": cancer}
