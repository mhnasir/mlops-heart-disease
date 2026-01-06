
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import pickle

df = pd.read_csv("data/raw/heart.csv")

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(n_estimators=100)
}

for name, model in models.items():
    with mlflow.start_run(run_name=name):
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", model)
        ])
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        acc = accuracy_score(y_test, preds)
        roc = roc_auc_score(y_test, preds)

        mlflow.log_param("model", name)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("roc_auc", roc)
        mlflow.sklearn.log_model(pipe, "model")

        if name == "RandomForest":
            pickle.dump(pipe, open("models/model.pkl", "wb"))
