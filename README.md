# 🩺 Breast Cancer Prediction — Machine Learning + FastAPI

End-to-end ML project that predicts whether a breast tumor is likely to be **cancerous or not** using the Breast Cancer Wisconsin dataset.

## ✅ Features Included
- Full ML training pipeline (Logistic Regression)
- Cross-validation + hyperparameter tuning (GridSearchCV)
- Threshold tuning
- Random Forest comparison
- SHAP feature explainability
- Model & metadata saving (joblib + JSON)
- FastAPI backend for real-time inference
- Simple HTML/JS frontend that calls the API

---

## 📂 Project Structure

```
breast-cancer-machine-learning/
│
├── api/
│   └── app.py                      # FastAPI backend (loads saved model)
│
├── frontend/
│   └── index.html                  # Simple HTML + JS UI
│
├── breast_cancer_best_pipeline.joblib   # Saved sklearn pipeline (scaler + model)
├── breast_cancer_metadata.json          # Metadata (features, hyperparams, threshold)
├── main.py                             # Training script (full ML pipeline)
├── requirements.txt
└── README.md
```

> Note: `.venv/` and other local environment files should be ignored via `.gitignore`.

---

## 📊 Dataset

Using `sklearn.datasets.load_breast_cancer`:

- **Samples**: 569  
- **Features**: 30 numeric features per tumor  
- **Target labels**:  
  - `0` → Malignant (**cancer**)  
  - `1` → Benign (**no cancer**)  

---

## 🛠️ Setup & Installation

### 1️⃣ Create and activate virtual environment
```
python3 -m venv .venv
source .venv/bin/activate
```

Windows (PowerShell):
```
.venv\Scripts\Activate
```

### 2️⃣ Install dependencies
```
pip install -r requirements.txt
```

Minimal `requirements.txt`:
```
fastapi
uvicorn
scikit-learn
pandas
numpy
joblib
matplotlib
shap
```

---

## 🚀 Run the FastAPI Backend

From the **api/** folder:

```
cd api
uvicorn app:app --reload
```

API will start at:

- Base URL: http://127.0.0.1:8000  
- Swagger docs: http://127.0.0.1:8000/docs

Terminal output:

```
Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

---

## 🌐 Run the Frontend

Open:

```
frontend/index.html
```

Either double-click OR use VS Code Live Server.

The UI allows you to:

- Paste 30 comma-separated feature values  
- Click **Predict**
- See results like:

**YES — Breast cancer detected**  
or  
**NO — Breast cancer NOT detected**

---

## 🔌 API — Endpoint Details

### POST `/predict`

URL:

```
http://127.0.0.1:8000/predict
```

### Request body:
```
{
  "data": [
    17.99, 10.38, 122.80, 1001.0, 0.1184,
    0.2776, 0.3001, 0.1471, 0.2419, 0.07871,
    1.095, 0.9053, 8.589, 153.4, 0.006399,
    0.04904, 0.05373, 0.01587, 0.03003,
    0.006193, 25.38, 17.33, 184.6, 2019.0,
    0.1622, 0.6656, 0.7119, 0.2654, 0.4601,
    0.1189
  ]
}
```

### Response:
```
{ "cancer": true }
```
or
```
{ "cancer": false }
```

Meaning:

- `true` → Breast cancer detected (**malignant**)  
- `false` → Breast cancer NOT detected (**benign**)  

---

## 🧠 Training Script (main.py)

Workflow (summarized):

- Load dataset (`load_breast_cancer`)
- Build X (features) and y (labels)
- Train/test split - stratified
- Pipeline → StandardScaler + LogisticRegression
- Cross-validation (StratifiedKFold)
- Hyperparameter tuning (GridSearchCV)
- Metrics: Accuracy, Precision, Recall, F1
- SHAP explainability
- RandomForest comparison
- Threshold tuning for best F1
- Save:
  - `breast_cancer_best_pipeline.joblib`
  - `breast_cancer_metadata.json`

---

## 📈 Example Model Performance

| Metric     | Score |
|-----------|-------|
| Accuracy  | ~0.98 |
| Precision | ~0.99 |
| Recall    | ~0.99 |
| F1-score  | ~0.99 |
| Mean CV   | ~0.97 |

---

## 📦 Saved Artifacts

### `breast_cancer_best_pipeline.joblib`
- Full pipeline (scaler + tuned model)

### `breast_cancer_metadata.json`
- Feature names  
- Best hyperparameters  
- Custom threshold  
- Version info  

Used directly by FastAPI.

---

## 🧪 Test the API using cURL

```
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"data":[17.99,10.38,122.80,1001.0,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871,1.095,0.9053,8.589,153.4,0.006399,0.04904,0.05373,0.01587,0.03003,0.006193,25.38,17.33,184.6,2019.0,0.1622,0.6656,0.7119,0.2654,0.4601,0.1189]}'
```

---

## 🚀 Next Steps

- Dockerize backend  
- Deploy API (Render / Railway / AWS / GCP)  
- Deploy frontend (GitHub Pages / Vercel)  
- Add logging + monitoring  
- Add unit tests
- Improve UI

---

