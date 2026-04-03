import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def run_eda(input_path):
    os.makedirs("Data/eda", exist_ok=True)

    # load data
    df = pd.read_csv(input_path)

    plt.figure(figsize=(6,4))
    sns.countplot(x="num", data=df)
    plt.title("Heart Disease Risk Distribution")
    plt.savefig("Data/eda/target_distribution.png")
    plt.close()

    plt.figure(figsize=(12,8))
    numeric_df = df.select_dtypes(include=['number'])
    sns.heatmap(numeric_df.corr(), cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.savefig("Data/eda/correlation_heatmap.png")
    plt.close()

    print("EDA Complete")