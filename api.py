from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from mlp import MLP

app = FastAPI(
    title="Numba‑MLP API",
    version="1.1.0",
    description="API d'entraînement et de prédiction via un réseau de neurones simple avec Numba (JSON via POST)."
)

# Instance globale du modèle
model: Optional[MLP] = None

# --- Schémas de requête et réponse ---

class TrainRequest(BaseModel):
    architecture: List[int]
    X: List[List[float]]
    y: List[List[float]]
    alpha: float = 0.1
    iterations: int = 10000

class TrainResponse(BaseModel):
    status: str
    iterations: int
    architecture: List[int]

class PredictRequest(BaseModel):
    inputs: List[float]

class PredictResponse(BaseModel):
    outputs: List[float]

# --- Routes API ---

@app.get("/")
def root():
    return {"message": "Bienvenue sur l'API Numba‑MLP. Accédez à /docs pour tester."}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/train", response_model=TrainResponse)
def train_model(req: TrainRequest):
    global model

    X_np = np.array(req.X, dtype=np.float64)
    y_np = np.array(req.y, dtype=np.float64)

    model = MLP(req.architecture)
    model.train(X_np, y_np, alpha=req.alpha, nb_iter=req.iterations)

    return TrainResponse(
        status="trained",
        iterations=req.iterations,
        architecture=req.architecture
    )

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    global model
    if model is None:
        raise HTTPException(status_code=400, detail="Model not trained yet. Call /train first.")

    if len(req.inputs) != model.d[0]:
        raise HTTPException(status_code=400, detail=f"Expected {model.d[0]} features, got {len(req.inputs)}")

    inputs_np = np.array(req.inputs, dtype=np.float64)
    outputs = model.predict(inputs_np)

    return PredictResponse(outputs=outputs.tolist())

@app.get("/ping")
def ping():
    return {"message": "pong"}
