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

model: Optional[MLP] = None

class TrainRequest(BaseModel):
    architecture: List[int]
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

@app.get("/")
def root():
    return {"message": "Bienvenue sur l'API Numba‑MLP. Accédez à /docs pour tester."}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/train", response_model=TrainResponse)
def train_model(req: TrainRequest):
    global model
    X = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y = [[0], [1], [1], [0]]

    model = MLP(req.architecture)
    model.train(np.array(X, dtype=np.float64), np.array(y, dtype=np.float64), alpha=req.alpha, nb_iter=req.iterations)

    return TrainResponse(status="trained", iterations=req.iterations, architecture=req.architecture)

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
