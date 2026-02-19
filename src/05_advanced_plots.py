import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier

X_train, X_test, y_train, y_test = joblib.load("data/processed.pkl")

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

lr = LogisticRegression(max_iter=1000)
rf = RandomForestClassifier(random_state=42)
xgb = XGBClassifier(eval_metric='logloss')

ensemble = VotingClassifier(
    estimators=[('lr', lr), ('rf', rf), ('xgb', xgb)],
    voting='soft',
    weights=[3,1,1]
)

ensemble.fit(X_train_res, y_train_res)

ConfusionMatrixDisplay.from_estimator(ensemble, X_test, y_test,cmap="Blues")
plt.title("AIHRM-CVD Confusion Matrix")
plt.savefig("outputs/plots/confusion_matrix.png", dpi=300)
plt.show()

RocCurveDisplay.from_estimator(ensemble, X_test, y_test)
plt.title("AIHRM-CVD ROC Curve")
plt.savefig("outputs/plots/roc_curve.png", dpi=300)
plt.show()
