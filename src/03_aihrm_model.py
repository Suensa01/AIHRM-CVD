import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

X_train, X_test, y_train, y_test = joblib.load("data/processed.pkl")


smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print("After SMOTE:", X_train_res.shape)

X_test_selected = X_test

lr = LogisticRegression(max_iter=1000)
rf = RandomForestClassifier(random_state=42)
xgb = XGBClassifier(eval_metric='logloss')

ensemble = VotingClassifier(
    estimators=[
        ('lr', lr),
        ('rf', rf),
        ('xgb', xgb)
    ],
    voting='soft',
    weights=[3,1,1]  
)
ensemble.fit(X_train_res, y_train_res)

joblib.dump(ensemble, "models/aihrm_cvd_model.pkl")
print("Model saved successfully.")
preds = ensemble.predict(X_test_selected)

results = {
    "Model": "AIHRM-CVD",
    "Accuracy": accuracy_score(y_test, preds),
    "Precision": precision_score(y_test, preds),
    "Recall": recall_score(y_test, preds),
    "F1": f1_score(y_test, preds)
}

results_df = pd.DataFrame([results])

results_df.to_csv("outputs/metrics/aihrm_results.csv", index=False)

print("\nAIHRM-CVD Results:\n")
print(results_df)
