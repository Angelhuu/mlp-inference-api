from typing import List, Optional
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
import numpy as np
from mlp import MLP

app = FastAPI(
    title="Numba‑MLP API",
    version="1.0.0",
    description="API d'entraînement et de prédiction via un réseau de neurones simple avec Numba."
)

# Modèle global (entraîné ou non)
model: Optional[MLP] = None

# --- Schemas ---
class TrainResponse(BaseModel):
    status: str
    iterations: int
    architecture: List[int]

class PredictResponse(BaseModel):
    outputs: List[float]

# --- Routes ---
@app.get("/")
def root():
    return {"message": "Bienvenue sur l'API Numba‑MLP. Accédez à /docs pour tester."}

@app.get("/favicon.ico")
def favicon():
    return {}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/train", response_model=TrainResponse)
def train_model(
    architecture: str = Query(..., description="Architecture, ex: 2,3,1"),
    alpha: float = 0.1,
    iterations: int = 10000
):
    global model

    arch_list = [int(x) for x in architecture.split(",")]
    X = [[0, 0], [0, 1], [1, 0], [1, 1]]
    y = [[0], [1], [1], [0]]

    model = MLP(arch_list)
    model.train(np.array(X, dtype=np.float64), np.array(y, dtype=np.float64), alpha=alpha, nb_iter=iterations)

    return TrainResponse(
        status="trained",
        iterations=iterations,
        architecture=arch_list
    )

@app.get("/predict", response_model=PredictResponse)
def predict(inputs: str = Query(..., description="Exemple : inputs=1.0,0.0")):
    global model
    if model is None:
        raise HTTPException(status_code=400, detail="Model not trained yet. Call /train first.")

    inputs_list = [float(x) for x in inputs.split(",")]
    if len(inputs_list) != model.d[0]:
        raise HTTPException(status_code=400, detail=f"Expected {model.d[0]} features, got {len(inputs_list)}")

    inputs_np = np.array(inputs_list, dtype=np.float64)
    outputs = model.predict(inputs_np)

    return PredictResponse(outputs=outputs.tolist())
