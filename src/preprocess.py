
import pandas as pd

def preprocess(df):
    df = df.drop_duplicates()
    df = df.fillna(df.median())
    return df
