"""Load Balaji's trained ensemble checkpoints and expose `load_model`, `predict`.

Expected layout (default):

  ML/checkpoints/artifacts/  tfidf_full.pkl, tfidf_diff.pkl, scaler.pkl
  ML/checkpoints/model/      final_sgd.pkl, final_nb.pkl, final_logreg.pkl,
                             ensemble_weights.npy, model_metadata.json

Environment overrides:

  GPT_ARTIFACTS_DIR  directory containing TF-IDF + scaler pickles
  GPT_MODEL_DIR      directory containing classifier pickles + ensemble weights

Example:

    from predict_model import load_model, predict
    model, _ = load_model()
    pred, probs = predict(model, None, prompt, resp_a, resp_b)
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from ensemble_engine import EnsemblePreferencePredictor

_SCRIPT_DIR = Path(__file__).resolve().parent


def default_artifact_dir() -> Path:
    env = os.getenv("GPT_ARTIFACTS_DIR")
    if env:
        return Path(env)
    return _SCRIPT_DIR / "checkpoints" / "artifacts"


def default_model_dir() -> Path:
    env = os.getenv("GPT_MODEL_DIR")
    if env:
        return Path(env)
    return _SCRIPT_DIR / "checkpoints" / "model"


def load_model(artifacts_dir: str | Path | None = None, model_dir: str | Path | None = None):
    """
    Returns ``(engine, engine)`` so existing Backend checks
    ``MODEL is None or VECT is None`` still work (both must be loaded).
    """
    ad = Path(artifacts_dir) if artifacts_dir is not None else default_artifact_dir()
    md = Path(model_dir) if model_dir is not None else default_model_dir()
    engine = EnsemblePreferencePredictor(ad, md)
    engine.load()
    return engine, engine


def predict(model, vectorizer, prompt: str, response_a: str, response_b: str):
    """
    Compatible with Backend/main.py legacy signature.

    ``vectorizer`` is unused but kept so imports do not change.
    """
    _ = vectorizer
    engine: EnsemblePreferencePredictor = model
    probs = engine.predict_proba(prompt, response_a, response_b)
    label_idx = int(np.argmax(probs))
    label_map = {0: "A", 1: "B", 2: "Tie"}
    return label_map[label_idx], probs


if __name__ == "__main__":
    eng, _ = load_model()
    demo_p = "Explain gradient descent briefly."
    demo_a = "Gradient descent minimizes loss by stepping opposite the gradient."
    demo_b = "It is an optimization trick."
    ph, pv = predict(eng, None, demo_p, demo_a, demo_b)
    print("pred:", ph, "probs:", pv)
