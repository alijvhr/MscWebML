import pandas as pd
from fastapi import HTTPException
from sklearn.preprocessing import LabelEncoder

from services.dataset import DatasetService


class EncodeService:
    def __init__(self):
        self.dataset_service = DatasetService()

    def encode(self, dataset_id: str, method: str = "onehot", target_column: str | None = None):
        file_path = self.dataset_service.get_path(dataset_id)
        df = pd.read_csv(file_path)
        categorical_columns = list(df.select_dtypes(include=["object", "category"]).columns)

        if target_column is not None and target_column not in df.columns:
            raise HTTPException(status_code=400, detail="target_column does not exist")

        target_was_categorical = target_column in categorical_columns
        feature_categorical_columns = [
            column for column in categorical_columns if column != target_column
        ]

        if method == "onehot":
            df = pd.get_dummies(df, columns=feature_categorical_columns)
        elif method == "label":
            encoder = LabelEncoder()
            for column in feature_categorical_columns:
                df[column] = encoder.fit_transform(df[column].astype(str))
        else:
            raise HTTPException(status_code=400, detail="Method must be 'onehot' or 'label'")

        if target_was_categorical:
            target_encoder = LabelEncoder()
            df[target_column] = target_encoder.fit_transform(df[target_column].astype(str))

        new_id = self.dataset_service.save_new_dataset(df, f"encoded_{dataset_id}.csv")
        return {
            "message": "Dataset encoded",
            "source_dataset_id": dataset_id,
            "new_dataset_id": new_id,
            "method": method,
            "encoded_columns": feature_categorical_columns,
            "target_column": target_column,
            "columns": list(df.columns),
        }
