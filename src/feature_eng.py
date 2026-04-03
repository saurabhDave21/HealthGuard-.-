import pandas as pd
import os

def feature_engineering(input_path, output_path):

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # load clean data
    df = pd.read_csv(input_path)


    # age × cholesterol
    if "age" in df.columns and "chol" in df.columns:
        df["age_chol"] = df["age"] * df["chol"]

    # blood pressure × heart rate
    if "trestbps" in df.columns and "thalach" in df.columns:
        df["bp_hr"] = df["trestbps"] * df["thalach"]

    # oldpeak × age
    if "oldpeak" in df.columns and "age" in df.columns:
        df["oldpeak_age"] = df["oldpeak"] * df["age"]

    df.to_csv(output_path, index=False)

    print("Feature engineering successful:", output_path)

    return output_path