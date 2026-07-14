# Machine Learning API with FastAPI

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
uvicorn main:app --reload
```

## Main Endpoints

- `POST /datasets/upload` upload a CSV
- `GET /datasets` list datasets
- `POST /preprocess` fill missing and normalize
- `POST /encode` one-hot or label encoding
- `POST /train` train model
- `GET /models` list trained models
- `GET /models/{model_id}` returns one saved model record with metrics
- `POST /predict` predicts using saved model.


## Algorithms

- `decision_tree`
- `svm`
- `knn`


## Example Train Body

```json
{
  "dataset_id": "DATASET_ID_HERE",
  "algorithm": "decision_tree",
  "target_column": "passed"
}
```

## Example Label Encoding Body

```json
{
  "dataset_id": "DATASET_ID_HERE",
  "method": "onehot",
  "target_column": "passed"
}
```

## Example Predict Body

```json
{
  "model_id": "MODEL_ID_HERE",
  "input_data": {
    "age": 20,
    "hours_studied": 6,
    "major": "CS"
  }
}
```
