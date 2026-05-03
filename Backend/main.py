import sys
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


Winner = Literal["model_a", "model_b", "tie"]


class LegacyPredictionRequest(BaseModel):
    prompt: str
    response_a: str
    response_b: str


class PredictionRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    model_a: str = Field(..., min_length=1)
    model_b: str = Field(..., min_length=1)
    response_a: str = Field(..., min_length=1)
    response_b: str = Field(..., min_length=1)


class PredictionResponse(BaseModel):
    winner: Winner
    winner_model: str | None
    confidence: float
    label: str
    scores: dict[str, float]


app = FastAPI(title="Global Preference Tester API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BACKEND_DIR.parent
FRONTEND_DIR = PROJECT_DIR / "Frontend"
ML_DIR = PROJECT_DIR / "ML"

app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

MODEL_OPTIONS = [
    "alpaca-13b",
    "chatglm-6b",
    "chatglm2-6b",
    "chatglm3-6b",
    "claude-1",
    "claude-2.0",
    "claude-2.1",
    "claude-instant-1",
    "codellama-34b-instruct",
    "deepseek-llm-67b-chat",
    "dolly-v2-12b",
    "dolphin-2.2.1-mistral-7b",
    "falcon-180b-chat",
    "fastchat-t5-3b",
    "gemini-pro",
    "gemini-pro-dev-api",
    "gpt-3.5-turbo-0125",
    "gpt-3.5-turbo-0314",
    "gpt-3.5-turbo-0613",
    "gpt-3.5-turbo-1106",
    "gpt-4-0125-preview",
    "gpt-4-0314",
    "gpt-4-0613",
    "gpt-4-1106-preview",
    "gpt4all-13b-snoozy",
    "guanaco-33b",
    "koala-13b",
    "llama-13b",
    "llama-2-13b-chat",
    "llama-2-70b-chat",
    "llama-2-7b-chat",
    "llama2-70b-steerlm-chat",
    "mistral-7b-instruct",
    "mistral-7b-instruct-v0.2",
    "mistral-medium",
    "mixtral-8x7b-instruct-v0.1",
    "mpt-30b-chat",
    "mpt-7b-chat",
    "nous-hermes-2-mixtral-8x7b-dpo",
    "oasst-pythia-12b",
    "openchat-3.5",
    "openchat-3.5-0106",
    "openhermes-2.5-mistral-7b",
    "palm-2",
    "pplx-70b-online",
    "pplx-7b-online",
    "qwen-14b-chat",
    "qwen1.5-4b-chat",
    "qwen1.5-72b-chat",
    "qwen1.5-7b-chat",
    "RWKV-4-Raven-14B",
    "solar-10.7b-instruct-v1.0",
    "stablelm-tuned-alpha-7b",
    "starling-lm-7b-alpha",
    "stripedhyena-nous-7b",
    "tulu-2-dpo-70b",
    "vicuna-13b",
    "vicuna-33b",
    "vicuna-7b",
    "wizardlm-13b",
    "wizardlm-70b",
    "yi-34b-chat",
    "zephyr-7b-alpha",
    "zephyr-7b-beta",
]


def predict_with_ml_model(payload: PredictionRequest) -> PredictionResponse:
    """Fallback heuristic used by the frontend API."""
    response_a_words = len(payload.response_a.split())
    response_b_words = len(payload.response_b.split())

    if abs(response_a_words - response_b_words) <= 2:
        return PredictionResponse(
            winner="tie",
            winner_model=None,
            confidence=0.5,
            label="Tied! Both models are equal!",
            scores={"model_a": 0.5, "model_b": 0.5},
        )

    total_words = response_a_words + response_b_words
    model_a_score = response_a_words / total_words
    model_b_score = response_b_words / total_words

    if model_a_score > model_b_score:
        return PredictionResponse(
            winner="model_a",
            winner_model=payload.model_a,
            confidence=round(model_a_score, 4),
            label=f"Model A is better: {payload.model_a}",
            scores={"model_a": round(model_a_score, 4), "model_b": round(model_b_score, 4)},
        )

    return PredictionResponse(
        winner="model_b",
        winner_model=payload.model_b,
        confidence=round(model_b_score, 4),
        label=f"Model B is better: {payload.model_b}",
        scores={"model_a": round(model_a_score, 4), "model_b": round(model_b_score, 4)},
    )


def build_prediction_response(payload: PredictionRequest) -> PredictionResponse:
    """Use the trained ensemble when loaded; otherwise word-count heuristic."""
    if MODEL is None or VECT is None:
        return predict_with_ml_model(payload)

    label, probs = predict(MODEL, VECT, payload.prompt, payload.response_a, payload.response_b)
    model_a_score = round(float(probs[0]), 4)
    model_b_score = round(float(probs[1]), 4)
    tie_score = round(float(probs[2]), 4)

    if label == "A":
        return PredictionResponse(
            winner="model_a",
            winner_model=payload.model_a,
            confidence=model_a_score,
            label=f"Model A is better: {payload.model_a}",
            scores={"model_a": model_a_score, "model_b": model_b_score, "tie": tie_score},
        )

    if label == "B":
        return PredictionResponse(
            winner="model_b",
            winner_model=payload.model_b,
            confidence=model_b_score,
            label=f"Model B is better: {payload.model_b}",
            scores={"model_a": model_a_score, "model_b": model_b_score, "tie": tie_score},
        )

    return PredictionResponse(
        winner="tie",
        winner_model=None,
        confidence=tie_score,
        label="Tied! Both models are equal!",
        scores={"model_a": model_a_score, "model_b": model_b_score, "tie": tie_score},
    )


if str(ML_DIR) not in sys.path:
    sys.path.append(str(ML_DIR))

try:
    from predict_model import load_model, predict  # type: ignore

    MODEL, VECT = load_model()
except Exception as exc:
    MODEL = None
    VECT = None
    _load_error = str(exc)
else:
    _load_error = None


@app.get("/")
async def root():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/api/status")
async def status():
    return {"status": "ok"}


@app.get("/api/models")
async def models():
    return {"models": MODEL_OPTIONS}


@app.post("/api/predict", response_model=PredictionResponse)
async def api_predict(payload: PredictionRequest):
    if payload.model_a.strip() == payload.model_b.strip():
        raise HTTPException(status_code=400, detail="Model A and Model B must be different.")

    try:
        return build_prediction_response(payload)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/health")
async def health():
    meta = getattr(MODEL, "metadata", None) if MODEL is not None else None
    return {
        "model_loaded": MODEL is not None,
        "predictor": "ensemble_sgd_nb_logreg" if MODEL is not None else None,
        "error": _load_error,
        "eval_metrics": (
            {
                "log_loss": meta.get("eval_log_loss"),
                "macro_f1": meta.get("eval_macro_f1"),
                "accuracy": meta.get("eval_accuracy"),
            }
            if isinstance(meta, dict)
            else None
        ),
    }


@app.get("/api/model-info")
async def model_info():
    """Training metadata from ``model_metadata.json`` (when ensemble is loaded)."""
    if MODEL is None:
        return {"loaded": False, "error": _load_error}
    return {"loaded": True, "metadata": getattr(MODEL, "metadata", {})}


@app.post("/predict")
async def predict_endpoint(payload: LegacyPredictionRequest) -> Any:
    """Return preferred response (A / B / Tie) and class probabilities."""
    if MODEL is None or VECT is None:
        raise HTTPException(status_code=500, detail={"error": "model not loaded", "reason": _load_error})

    try:
        label, probs = predict(MODEL, VECT, payload.prompt, payload.response_a, payload.response_b)
        probs_list = {
            "winner_model_a": float(probs[0]),
            "winner_model_b": float(probs[1]),
            "winner_tie": float(probs[2]),
        }
        return JSONResponse({"prediction": label, "probabilities": probs_list})
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
