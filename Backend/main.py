from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
import sys
from typing import Any, Dict
from fastapi import HTTPException
from fastapi.responses import JSONResponse


class dataformat(BaseModel):
    prompt: str
    response_a: str
    response_b: str

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent

# Make sure the ML folder is importable so we can load the saved artifacts
ml_path = BASE_DIR.parent / "ML"
if str(ml_path) not in sys.path:
    sys.path.append(str(ml_path))

try:
    from predict_model import load_model, predict  # type: ignore
    MODEL, VECT = load_model()
except Exception as e:
    MODEL = None
    VECT = None
    _load_error = str(e)
else:
    _load_error = None


@app.get("/")
async def root():
    return FileResponse(BASE_DIR / "Frontend" / "index.html")


@app.get("/health")
async def health():
    return {"model_loaded": MODEL is not None, "error": _load_error}


@app.post("/predict")
async def predict_endpoint(payload: dataformat) -> Any:
    """Return preferred response (A / B / Tie) and class probabilities.

    JSON body: {"prompt": "..", "response_a": "..", "response_b": ".."}
    """
    if MODEL is None or VECT is None:
        raise HTTPException(status_code=500, detail={"error": "model not loaded", "reason": _load_error})

    try:
        label, probs = predict(MODEL, VECT, payload.prompt, payload.response_a, payload.response_b)
        # probs is a numpy array; convert to floats
        probs_list = {
            "winner_model_a": float(probs[0]),
            "winner_model_b": float(probs[1]),
            "winner_tie": float(probs[2]),
        }
        return JSONResponse({"prediction": label, "probabilities": probs_list})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))