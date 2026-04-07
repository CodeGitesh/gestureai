import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

df = pd.read_csv("data/gestures.csv", header=None)

X = df.iloc[:, :-1]
y = df.iloc[:, -1]

model = LogisticRegression(max_iter=2000)
model.fit(X, y)

joblib.dump(model, "gesture_model.pkl")
print("Model trained & saved")
