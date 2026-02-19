import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

df = pd.read_csv("data/HeartDiseaseTrain-Test.csv")
df = df.drop_duplicates()

print("Dataset shape after removing duplicates:", df.shape)
df = df.dropna()
X = df.drop("target", axis=1)
y = df["target"]
X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

joblib.dump(scaler, "models/scaler.pkl")
joblib.dump((X_train, X_test, y_train, y_test), "data/processed.pkl")

print("Preprocessing completed.")
print("Final shape after encoding:", X_train.shape)
