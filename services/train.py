import hashlib
import json
import pickle
from datetime import datetime

import pandas as pd
from fastapi import HTTPException
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from algorithms.decision_tree import create_decision_tree
from algorithms.knn import create_knn
from algorithms.svm import create_svm
from services.database import MODELS_DIR, get_connection
from services.dataset import DatasetService


class TrainService:
    def __init__(self):
        self.dataset_service = DatasetService()

    def train(
        self,
        dataset_id: str,
        algorithm: str,
        target_column: str | None = None,
        feature_columns: list[str] | None = None,
    ):
        algorithm = algorithm.lower().replace("-", "_")
        file_path = self.dataset_service.get_path(dataset_id)
        df = pd.read_csv(file_path)

        if not target_column:
            raise HTTPException(status_code=400, detail="target_column is required")
        model, features, metrics = self._train_supervised(df, algorithm, target_column, feature_columns)

        model_id = hashlib.sha256(f"{dataset_id}{algorithm}{datetime.utcnow().isoformat()}".encode()).hexdigest()[:16]
        model_path = MODELS_DIR / f"{model_id}.pkl"

        saved_data = {
            "model": model,
            "algorithm": algorithm,
            "feature_columns": features,
            "target_column": target_column,
            "metrics": metrics,
        }
        with open(model_path, "wb") as model_file:
            pickle.dump(saved_data, model_file)

        with get_connection() as conn:
            conn.execute(
                """
                INSERT INTO models
                (
                    id, algorithm, dataset_id, target_column, feature_columns,
                    file_path, score, accuracy, precision, sensitivity,
                    specificity, f1_score, confusion_matrix, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    model_id,
                    algorithm,
                    dataset_id,
                    target_column,
                    json.dumps(features),
                    str(model_path),
                    metrics["score"],
                    metrics["accuracy"],
                    metrics["precision"],
                    metrics["sensitivity"],
                    metrics["specificity"],
                    metrics["f1_score"],
                    json.dumps(metrics["confusion_matrix"]),
                    datetime.utcnow().isoformat(),
                ),
            )
            conn.commit()

        return {
            "message": "Model trained",
            "model_id": model_id,
            "algorithm": algorithm,
            "dataset_id": dataset_id,
            "feature_columns": features,
            "target_column": target_column,
            "metrics": metrics,
        }

    def _train_supervised(self, df, algorithm, target_column, feature_columns):
        if target_column not in df.columns:
            raise HTTPException(status_code=400, detail="target_column does not exist")

        if feature_columns is None:
            feature_columns = [column for column in df.columns if column != target_column]

        missing = [column for column in feature_columns if column not in df.columns]
        if missing:
            raise HTTPException(status_code=400, detail=f"Feature columns not found: {missing}")

        x = pd.get_dummies(df[feature_columns])
        y = df[target_column]
        features = list(x.columns)

        if len(df) < 2:
            raise HTTPException(status_code=400, detail="Dataset needs at least 2 rows")

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
        model, is_classification = self._create_model(algorithm, y)
        model.fit(x_train, y_train)
        score = float(model.score(x_test, y_test)) if len(x_test) > 0 else None
        metrics = self._calculate_metrics(model, x_test, y_test, score, is_classification)
        return model, features, metrics

    def _create_model(self, algorithm, y):
        is_classification = y.dtype == "object" or y.nunique() <= 10

        if algorithm == "decision_tree":
            return create_decision_tree(is_classification), is_classification
        if algorithm == "knn":
            return create_knn(is_classification), is_classification
        if algorithm == "svm":
            return create_svm(is_classification), is_classification

        raise HTTPException(
            status_code=400,
            detail="Algorithm must be decision_tree, svm, or knn",
        )

    def _calculate_metrics(self, model, x_test, y_test, score, is_classification):
        metrics = self._empty_metrics()
        metrics["score"] = score

        if not is_classification:
            return metrics

        y_pred = model.predict(x_test)
        labels = sorted(list(set(y_test) | set(y_pred)))
        matrix = confusion_matrix(y_test, y_pred, labels=labels)

        metrics["accuracy"] = float(accuracy_score(y_test, y_pred))
        metrics["precision"] = float(precision_score(y_test, y_pred, average="weighted", zero_division=0))
        metrics["sensitivity"] = float(recall_score(y_test, y_pred, average="weighted", zero_division=0))
        metrics["specificity"] = self._calculate_specificity(matrix)
        metrics["f1_score"] = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))
        metrics["confusion_matrix"] = {
            "labels": [str(label) for label in labels],
            "matrix": matrix.tolist(),
        }
        return metrics

    def _calculate_specificity(self, matrix):
        specificities = []
        total = matrix.sum()

        for index in range(len(matrix)):
            true_negative = total - matrix[index, :].sum() - matrix[:, index].sum() + matrix[index, index]
            false_positive = matrix[:, index].sum() - matrix[index, index]
            denominator = true_negative + false_positive
            if denominator > 0:
                specificities.append(true_negative / denominator)

        if not specificities:
            return None
        return float(sum(specificities) / len(specificities))

    def _empty_metrics(self):
        return {
            "score": None,
            "accuracy": None,
            "precision": None,
            "sensitivity": None,
            "specificity": None,
            "f1_score": None,
            "confusion_matrix": None,
        }

    def list_all(self):
        with get_connection() as conn:
            rows = conn.execute(
                """
                SELECT
                    id, algorithm, dataset_id, target_column, feature_columns,
                    score, accuracy, precision, sensitivity, specificity,
                    f1_score, confusion_matrix, created_at
                FROM models
                ORDER BY created_at DESC
                """
            ).fetchall()
        return {"models": [self._format_model_row(row) for row in rows]}

    def get_by_id(self, model_id: str):
        with get_connection() as conn:
            row = conn.execute(
                """
                SELECT
                    id, algorithm, dataset_id, target_column, feature_columns,
                    file_path, score, accuracy, precision, sensitivity,
                    specificity, f1_score, confusion_matrix, created_at
                FROM models
                WHERE id = ?
                """,
                (model_id,),
            ).fetchone()

        if row is None:
            raise HTTPException(status_code=404, detail="Model not found")

        return self._format_model_row(row, include_file_path=True)

    def _format_model_row(self, row, include_file_path: bool = False):
        data = dict(row)
        data["feature_columns"] = json.loads(data["feature_columns"]) if data["feature_columns"] else []
        data["confusion_matrix"] = (
            json.loads(data["confusion_matrix"]) if data["confusion_matrix"] else None
        )
        data["metrics"] = {
            "score": data.pop("score"),
            "accuracy": data.pop("accuracy"),
            "precision": data.pop("precision"),
            "sensitivity": data.pop("sensitivity"),
            "specificity": data.pop("specificity"),
            "f1_score": data.pop("f1_score"),
            "confusion_matrix": data.pop("confusion_matrix"),
        }
        if not include_file_path:
            data.pop("file_path", None)
        return data
