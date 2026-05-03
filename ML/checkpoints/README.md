# Checkpoints (not in Git — too large for GitHub)

Copy your trained files here so inference works:

```
artifacts/
  tfidf_full.pkl
  tfidf_diff.pkl
  scaler.pkl
model/
  final_sgd.pkl
  final_nb.pkl
  final_logreg.pkl
  ensemble_weights.npy
  model_metadata.json
```

**From your llm_preference_predictor repo** (same filenames under `artifacts/` and `model/`):

```powershell
$src = "C:\path\to\llm_preference_predictor"
$dst = ".\ML\checkpoints"
Copy-Item "$src\artifacts\*.pkl"      "$dst\artifacts\"
Copy-Item "$src\model\*.pkl","$src\model\*.npy","$src\model\model_metadata.json" "$dst\model\"
```

Or use **[Git LFS](https://git-lfs.github.com)** if your team prefers binaries in Git.

Custom paths at runtime:

- `GPT_ARTIFACTS_DIR` → folder containing the three artifact `.pkl` files  
- `GPT_MODEL_DIR` → folder containing the five model files  
