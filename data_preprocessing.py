import re
import pandas as pd


def preprocess_fun(text):
    if isinstance(text, str):
        # Tokenization
        tokens = re.findall(r"\b\w+\b", text.lower())
        # Stop-word removal and stemming can be added here if needed
        return " ".join(tokens)
    else:
        return "doctor"


def read_data(data_path):
    data = pd.read_csv(data_path, encoding="latin1")
    # Read data from CSV files
    business_d = pd.read_csv("business_d.csv",  encoding="latin1")
    food_d = pd.read_csv("food_d.csv",  encoding="latin1")
    health_d = pd.read_csv("health_d.csv",  encoding="latin1")
    totaldata = pd.concat([business_d, food_d, health_d], ignore_index=True)
    totaldata["text"] = totaldata["text"].apply(preprocess_fun)
    return totaldata
