import pandas as pd

df = pd.read_csv("data/HeartDiseaseTrain-Test.csv")

print(df.head())
print("\nColumns:\n", df.columns)
print("\nDuplicate rows:", df.duplicated().sum())
