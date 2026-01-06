
import pandas as pd

def load_and_preprocess(path):
    df = pd.read_csv(path)
    df = df.drop_duplicates()
    df = df.fillna(df.median(numeric_only=True))
    X = df.drop("target", axis=1)
    y = df["target"]
    return X, y
