"""Minimal trainer: trains a TF-IDF + LogisticRegression and saves artifacts.
Usage: run from anywhere; scripts resolve paths relative to their file.
"""
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib


def main():
    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir / "llm-classification-finetuning"
    train_path = data_dir / "train.csv"

    df = pd.read_csv(train_path)

    # Build input text per example: include prompt and both responses
    texts = (
        df['prompt'].fillna('')
        + " [A] " + df['response_a'].fillna('')
        + " [B] " + df['response_b'].fillna('')
    )

    # Labels: winner columns -> argmax
    winner_cols = [c for c in df.columns if 'winner' in c.lower()]
    probs = df[winner_cols].values
    y = probs.argmax(axis=1)

    # Vectorize
    vec = TfidfVectorizer(max_features=2000, ngram_range=(1,2))
    X = vec.fit_transform(texts)

    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_tr, y_tr)

    y_pred = model.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    print(f"Validation accuracy: {acc:.4f}")

    # Save artifacts
    out_dir = data_dir / "artifacts"
    out_dir.mkdir(exist_ok=True)
    joblib.dump(model, out_dir / "model.joblib")
    joblib.dump(vec, out_dir / "vectorizer.joblib")
    print(f"Saved model and vectorizer to: {out_dir}")


if __name__ == '__main__':
    main()
