
# Fully Working MLOps Heart Disease Project - Assingnment Group 
CI/CD - Full Implemetation
 Sem 3 Group 86 
## Setup
pip install -r requirements.txt

## Train & Track
mlflow ui
python src/train.py

## Test
pytest

## API
uvicorn app.main:app --reload
http://localhost:8000/docs

## Docker
docker build -t heart-api .
docker run -p 8000:8000 heart-api
=======
 4dd9f57bd3b0bf8c75d379c5a63329d38bfdef79
