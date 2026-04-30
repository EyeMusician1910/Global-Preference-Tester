"""
LLM Classification Finetuning - Basic Solution
Predicts which response humans prefer: A / B / Tie
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ==================== CONFIG ====================
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR / "llm-classification-finetuning"
TRAIN_FILE = DATA_DIR / "train.csv"
TEST_FILE = DATA_DIR / "test.csv"
SUBMISSION_FILE = DATA_DIR / "submission.csv"

# ==================== LOAD DATA ====================
print("Loading data...")
train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")
print(f"\nTrain columns: {train_df.columns.tolist()}")
print(f"Test columns: {test_df.columns.tolist()}")

# ==================== DATA EXPLORATION ====================
print("\n" + "="*50)
print("DATA EXPLORATION")
print("="*50)

# Check for label columns
label_cols = [col for col in train_df.columns if 'winner' in col.lower()]
print(f"Label columns: {label_cols}")

if label_cols:
    # Multi-class labels
    y = train_df[label_cols[0]]
    print(f"Unique labels: {y.unique()}")
    print(f"\nClass distribution:")
    print(y.value_counts())
else:
    print("No label columns found!")
    exit(1)

print(f"\nSample prompt: {train_df['prompt'].iloc[0][:100]}...")
print(f"Sample response_a: {train_df['response_a'].iloc[0][:100]}...")
print(f"Sample response_b: {train_df['response_b'].iloc[0][:100]}...")

# ==================== FEATURE ENGINEERING ====================
print("\n" + "="*50)
print("FEATURE ENGINEERING")
print("="*50)

def extract_features(df):
    """Extract features from text"""
    features = pd.DataFrame(index=df.index)
    
    # Length features
    features['prompt_len'] = df['prompt'].fillna("").astype(str).apply(len)
    features['response_a_len'] = df['response_a'].fillna("").astype(str).apply(len)
    features['response_b_len'] = df['response_b'].fillna("").astype(str).apply(len)
    
    # Word count features
    features['response_a_words'] = df['response_a'].fillna("").astype(str).apply(lambda x: len(x.split()))
    features['response_b_words'] = df['response_b'].fillna("").astype(str).apply(lambda x: len(x.split()))
    
    # Length difference (bias detection)
    features['len_diff'] = features['response_a_len'] - features['response_b_len']
    features['words_diff'] = features['response_a_words'] - features['response_b_words']
    
    return features

train_features = extract_features(train_df)
test_features = extract_features(test_df)

print("Hand-crafted features:")
print(train_features.describe())

# ==================== TEXT VECTORIZATION ====================
print("\n" + "="*50)
print("TEXT VECTORIZATION (TF-IDF)")
print("="*50)

# Combine prompts and responses for context
train_df['combined_a'] = train_df['prompt'].fillna("") + " " + train_df['response_a'].fillna("")
train_df['combined_b'] = train_df['prompt'].fillna("") + " " + train_df['response_b'].fillna("")
test_df['combined_a'] = test_df['prompt'].fillna("") + " " + test_df['response_a'].fillna("")
test_df['combined_b'] = test_df['prompt'].fillna("") + " " + test_df['response_b'].fillna("")

# TF-IDF for response A
vectorizer_a = TfidfVectorizer(max_features=100, ngram_range=(1, 2), min_df=2)
tfidf_a_train = vectorizer_a.fit_transform(train_df['combined_a']).toarray()
tfidf_a_test = vectorizer_a.transform(test_df['combined_a']).toarray()

# TF-IDF for response B
vectorizer_b = TfidfVectorizer(max_features=100, ngram_range=(1, 2), min_df=2)
tfidf_b_train = vectorizer_b.fit_transform(train_df['combined_b']).toarray()
tfidf_b_test = vectorizer_b.transform(test_df['combined_b']).toarray()

# Difference between A and B TF-IDF features
tfidf_diff_train = tfidf_a_train - tfidf_b_train
tfidf_diff_test = tfidf_a_test - tfidf_b_test

print(f"TF-IDF shape: {tfidf_a_train.shape}")
print(f"TF-IDF difference shape: {tfidf_diff_train.shape}")

# ==================== COMBINE FEATURES ====================
X_train = np.hstack([
    train_features.values,
    tfidf_a_train,
    tfidf_b_train,
    tfidf_diff_train
])

X_test = np.hstack([
    test_features.values,
    tfidf_a_test,
    tfidf_b_test,
    tfidf_diff_test
])

print(f"Final training feature shape: {X_train.shape}")
print(f"Final test feature shape: {X_test.shape}")

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==================== PREPARE LABELS ====================
# Find the winner column
winner_col = [col for col in train_df.columns if 'winner_model_a' in col.lower()]
if winner_col:
    y = train_df[winner_col[0]]
    # If probabilities, convert to class labels
    if y.dtype == float and y.max() <= 1.0:
        # This might be probability format, need to check actual label format
        print("Labels appear to be probabilities")
        # For now, assume it's a multi-output scenario
else:
    print("Could not find winner column!")
    print(f"Available columns: {train_df.columns.tolist()}")
    exit(1)

# ==================== TRAIN/VALIDATION SPLIT ====================
print("\n" + "="*50)
print("MODEL TRAINING")
print("="*50)

# Check if we have actual binary/multiclass labels
winner_cols = [col for col in train_df.columns if 'winner' in col.lower()]
if len(winner_cols) == 3:
    # Multi-output classification (winner_a, winner_b, winner_tie)
    # Create single label based on which is highest
    probs = train_df[winner_cols].values
    y = np.argmax(probs, axis=1)  # 0=a, 1=b, 2=tie
    label_map = {0: 'A', 1: 'B', 2: 'Tie'}
    print(f"Label distribution: {pd.Series(y).value_counts()}")
else:
    print("Unexpected label format")
    exit(1)

# Train/val split for validation
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Train model
print("Training Logistic Regression classifier...")
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_tr, y_tr)

# Validate
y_pred = model.predict(X_val)
val_accuracy = accuracy_score(y_val, y_pred)
print(f"\nValidation Accuracy: {val_accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_val, y_pred, target_names=['A', 'B', 'Tie']))

# ==================== PREDICTIONS ====================
print("\n" + "="*50)
print("GENERATING SUBMISSION")
print("="*50)

# Get probability predictions
y_test_proba = model.predict_proba(X_test_scaled)

# Create submission dataframe
submission = pd.DataFrame({
    'id': test_df['id'],
    'winner_model_a': y_test_proba[:, 0],
    'winner_model_b': y_test_proba[:, 1],
    'winner_tie': y_test_proba[:, 2]
})

print(submission.head())
print(f"\nSubmission shape: {submission.shape}")

# Save submission
submission.to_csv(SUBMISSION_FILE, index=False)
print(f"\n✅ Submission saved to: {SUBMISSION_FILE}")

# ==================== SUMMARY ====================
print("\n" + "="*50)
print("SUMMARY")
print("="*50)
print(f"Train samples: {len(train_df)}")
print(f"Test samples: {len(test_df)}")
print(f"Features used: {X_train.shape[1]}")
print(f"Validation accuracy: {val_accuracy:.4f}")
print(f"Model: Logistic Regression (multinomial)")
print(f"Classes: A, B, Tie")


#Accuracy is around 0.45, which is better than random guessing (0.33) but still leaves room for improvement.
