
import mlflow
import mlflow.sklearn
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Heart-Disease-Classification")

import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from preprocess import load_and_preprocess

X, y = load_and_preprocess("data/raw/heart.csv")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42)
}

best_auc = 0
best_model = None

for name, model in models.items():
    with mlflow.start_run(run_name=name):
        pipeline = Pipeline([("scaler", StandardScaler()), ("model", model)])
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        probs = pipeline.predict_proba(X_test)[:,1]
        acc = accuracy_score(y_test, preds)
        auc = roc_auc_score(y_test, probs)
        mlflow.log_param("model", name)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("roc_auc", auc)
        mlflow.sklearn.log_model(pipeline, "model")
        if auc > best_auc:
            best_auc = auc
            best_model = pipeline

joblib.dump(best_model, "models/model.pkl")
print("Best model ROC-AUC:", best_auc)
