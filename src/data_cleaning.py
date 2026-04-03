import pandas as pd
import os

def clean_data(input_path, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df = pd.read_csv(input_path)

    df.columns = df.columns.str.strip()

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])

    if "num" in df.columns:
        df["num"] = df["num"].apply(lambda x: 1 if x > 0 else 0)

    df.to_csv(output_path, index=False)

    print("Data cleaning completed",output_path)
    return output_path