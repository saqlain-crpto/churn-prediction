import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score

# Same cleaning and split as train.py (same random_state = same test set)
df = pd.read_csv("data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df = df.drop(columns=["customerID"])
df["Churn"] = (df["Churn"] == "Yes").astype(int)

X = df.drop(columns="Churn")
y = df["Churn"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

model = joblib.load("models/churn_model.joblib")
proba = model.predict_proba(X_test)[:, 1]

# 1. Full report (now with the churn row)
print(classification_report(y_test, model.predict(X_test)))

# 2. Threshold tuning
print("Threshold | Precision | Recall | F1")
for t in [0.3, 0.4, 0.5, 0.6, 0.7]:
    pred = (proba >= t).astype(int)
    print(f"{t:>9} | {precision_score(y_test, pred):.2f}      | "
          f"{recall_score(y_test, pred):.2f}   | {f1_score(y_test, pred):.2f}")

# 3. Top 10 features
names = model.named_steps["prep"].get_feature_names_out()
importances = model.named_steps["model"].feature_importances_
top = pd.Series(importances, index=names).sort_values(ascending=False).head(10)
print("\nTop 10 features:")
print(top)