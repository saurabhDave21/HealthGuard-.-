import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

def train_ml_model(input_path):

    os.makedirs("models", exist_ok=True)

    # we need to load dataset
    df = pd.read_csv(input_path)

    df.columns = df.columns.str.strip()
    df = pd.get_dummies(df, drop_first=True)
    X = df.drop("num", axis=1)
    y = df["num"]

    y = y.apply(lambda x: 1 if x > 0 else 0)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )


    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    #this is for check accuracy
    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)
    print(f"Model Accuracy: {acc * 100:.2f}%")
    with open("models/metrics.txt", "w") as f:f.write(f"Accuracy: {acc * 100:.2f}%")

    
    joblib.dump(model, "models/heart_model.pkl")
    joblib.dump(X.columns.tolist(), "models/columns.pkl")

    print("Model save 💯")

    return "models/heart_model.pkl"