import pandas as pd
import os
import joblib

def predict_ml(input_path):
    os.makedirs("data/predictions", exist_ok=True)

    #load data
    df = pd.read_csv(input_path)

    model = joblib.load("models/heart_model.pkl")
    columns = joblib.load("models/columns.pkl")
    df = pd.get_dummies(df)

    for col in columns:
        if col not in df:
            df[col] = 0
    X = df[columns]

    df["Prediction"] = model.predict(X)
    df["Risk"] = df["Prediction"].apply(lambda x: 1 if x == 1 else 0)

    df["Risk_Label"] = df["Prediction"].apply(
        lambda x: "High Risk" if x == 1 else "Low Risk"
    )

    #we save our prediction at this path
    output_path = "data/predictions/ml_predictions.csv"
    df.to_csv(output_path, index=False)

    print("Prediction saved:", output_path)