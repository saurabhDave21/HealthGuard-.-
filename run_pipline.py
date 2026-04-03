import pandas as pd
from src.data_cleaning import clean_data
from src.eda import run_eda
from src.feature_eng import feature_engineering
from src.train_ml_model import train_ml_model
from src.predict_ml import predict_ml


print("START PIPELINE")

#clean step1
cleaned = clean_data(
    "data/raw/cleanned.csv",
    "data/processed/patient_cleaned.csv"
)

#eda step2
run_eda(cleaned)

#featureEng.. step3
features = feature_engineering(
    cleaned,
    "data/processed/patient_features.csv"
)

#train step4
train_ml_model(features)

#predict steo5
predict_ml(features)


print("pipline finish")