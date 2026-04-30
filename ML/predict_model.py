"""Load trained artifacts and provide a `predict` function.
Example:
  from predict_model import load_model, predict
  model, vec = load_model()
  pred, probs = predict(model, vec, prompt, resp_a, resp_b)
"""
from pathlib import Path
import joblib
import numpy as np


def load_model(artifacts_dir: str | Path | None = None):
    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir / "llm-classification-finetuning"
    artifacts = Path(artifacts_dir) if artifacts_dir else data_dir / "artifacts"

    model = joblib.load(artifacts / "model.joblib")
    vectorizer = joblib.load(artifacts / "vectorizer.joblib")
    return model, vectorizer


def predict(model, vectorizer, prompt: str, response_a: str, response_b: str):
    text = (prompt or "") + " [A] " + (response_a or "") + " [B] " + (response_b or "")
    X = vectorizer.transform([text])
    probs = model.predict_proba(X)[0]
    label = int(np.argmax(probs))
    label_map = {0: 'A', 1: 'B', 2: 'Tie'}
    return label_map.get(label, str(label)), probs


if __name__ == '__main__':
    # quick demo using test.csv if present
    script_dir = Path(__file__).resolve().parent
    test_path = script_dir / "llm-classification-finetuning" / "test.csv"
    model, vec = load_model()
    if test_path.exists():
        import pandas as pd
        df = pd.read_csv(test_path)
        for i, row in df.head(5).iterrows():
            pred, probs = predict(model, vec, row['prompt'], row['response_a'], row['response_b'])
            print(row['id'], pred, probs)
    else:
        print('Artifacts loaded. Call predict(model, vectorizer, prompt, response_a, response_b)')
