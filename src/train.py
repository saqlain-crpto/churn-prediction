import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score

DATA_PATH = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"

# 1. Load and clean
df = pd.read_csv(DATA_PATH)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")  # blank strings become NaN
df = df.drop(columns=["customerID"])
df["Churn"] = (df["Churn"] == "Yes").astype(int)

print("Shape:", df.shape)
print("Churn rate:\n", df["Churn"].value_counts(normalize=True))

# 2. Split BEFORE preprocessing
X = df.drop(columns="Churn")
y = df["Churn"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 3. Preprocessing pipeline
num_cols = X.select_dtypes(include="number").columns.tolist()
cat_cols = X.select_dtypes(exclude="number").columns.tolist()

preprocess = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc", StandardScaler())]), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
])

# 4. Candidate models
models = {
    "LogReg": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "RandomForest": RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                           random_state=42, n_jobs=-1),
    "XGBoost": XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=4,
                             scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
                             eval_metric="logloss", random_state=42),
}

# 5. Compare with 5-fold cross-validation (ROC-AUC)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}
for name, model in models.items():
    pipe = Pipeline([("prep", preprocess), ("model", model)])
    scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
    results[name] = scores.mean()
    print(f"{name}: CV AUC = {scores.mean():.3f} (+/- {scores.std():.3f})")

best_name = max(results, key=results.get)
print("\nBest model:", best_name)

# 6. Final evaluation on the untouched test set
best = Pipeline([("prep", preprocess), ("model", models[best_name])])
best.fit(X_train, y_train)
pred = best.predict(X_test)
proba = best.predict_proba(X_test)[:, 1]
print(classification_report(y_test, pred))
print("Test ROC-AUC:", round(roc_auc_score(y_test, proba), 3))

# 7. Save the whole pipeline
os.makedirs("models", exist_ok=True)
joblib.dump(best, "models/churn_model.joblib")
print("Model saved to models/churn_model.joblib")