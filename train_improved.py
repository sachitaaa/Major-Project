from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

CSV = Path(__file__).parent / "final2" / "master_dataset_final.csv"
LABEL = "self_reported_difficulty"
ORDER = {"Easy": 0, "Medium": 1, "Hard": 2}

# Columns that must NOT be features (they leak the answer or come from it)
LEAKY = [
    "feature_score_raw",                  # feed feature_derived_difficulty
    "feature_score_scaled",
    "completion_time",                    # unclear origin: ask your friend
]

df = pd.read_csv(CSV, encoding="utf-8-sig")
print(f"Loaded {df.shape[0]} rows")

# ---------- Part 1: how much do the three difficulty labels agree? ----------
short = {
    "self_reported_difficulty": "self_reported",
    "feature_derived_difficulty": "feature_derived",
    "llm_response_augmented_difficulty": "llm_augmented",
}
nums = {short[c]: df[c].map(ORDER) for c in short}
names = list(nums)
agree = pd.DataFrame(index=names, columns=names, dtype=float)
for a in names:
    for b in names:
        agree.loc[a, b] = cohen_kappa_score(nums[a], nums[b], weights="quadratic")
print("\n=== Agreement between labels (quadratic weighted kappa) ===")
print(agree.round(2))

# ---------- Part 2: can text features predict the LLM's label? ----------
X = df.select_dtypes(include=["number", "bool"]).drop(columns=LEAKY, errors="ignore").astype(float)
y = df[LABEL].map(ORDER)
print(f"\nUsing {X.shape[1]} features:", X.columns.tolist())

models = {
    "baseline (always most common)": DummyClassifier(strategy="most_frequent"),
    "logistic regression (scaled)": make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)),
    "random forest": RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=42),
}

metric_names = ["accuracy", "f1_macro", "weighted_kappa"]
results = {name: [] for name in models}
for seed in range(5):                       # 5 different shuffles: small data is noisy
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for name, model in models.items():
        p = cross_val_predict(model, X, y, cv=skf)
        results[name].append([
            accuracy_score(y, p),
            f1_score(y, p, average="macro"),
            cohen_kappa_score(y, p, weights="quadratic"),
        ])

rows = {}
for name, r in results.items():
    r = np.array(r)
    rows[name] = {m: f"{r[:, i].mean():.3f} +/- {r[:, i].std():.3f}"
                  for i, m in enumerate(metric_names)}
print("\n=== Predicting", LABEL, "(mean +/- std over 5 shuffles) ===")
print(pd.DataFrame(rows).T)

rf = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42).fit(X, y)
print("\n=== Top 10 features (random forest) ===")
print(pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False).head(10).round(3))