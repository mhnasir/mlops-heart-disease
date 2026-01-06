
from src.preprocess import load_and_preprocess

def test_data_load():
    X, y = load_and_preprocess("data/raw/heart.csv")
    assert X.shape[0] > 0
    assert y.isnull().sum() == 0
