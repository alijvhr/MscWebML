import pandas as pd

from services.dataset import DatasetService


class PreprocessService:
    def __init__(self):
        self.dataset_service = DatasetService()

    def preprocess(self, dataset_id: str, fill_missing: bool = True, normalize: bool = True):
        file_path = self.dataset_service.get_path(dataset_id)
        df = pd.read_csv(file_path)

        if fill_missing:
            for column in df.columns:
                if pd.api.types.is_numeric_dtype(df[column]):
                    df[column] = df[column].fillna(df[column].mean())
                else:
                    mode = df[column].mode()
                    fill_value = mode[0] if not mode.empty else ""
                    df[column] = df[column].fillna(fill_value)

        if normalize:
            numeric_columns = df.select_dtypes(include=["number"]).columns
            for column in numeric_columns:
                minimum = df[column].min()
                maximum = df[column].max()
                if maximum != minimum:
                    df[column] = (df[column] - minimum) / (maximum - minimum)

        new_id = self.dataset_service.save_new_dataset(df, f"preprocessed_{dataset_id}.csv")
        return {
            "message": "Dataset preprocessed",
            "source_dataset_id": dataset_id,
            "new_dataset_id": new_id,
            "columns": list(df.columns),
        }
