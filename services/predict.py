import pickle

import pandas as pd
from fastapi import HTTPException

from services.database import get_connection


class PredictService:
    def predict(self, model_id: str, input_data):
        model_path = self._get_model_path(model_id)

        with open(model_path, "rb") as model_file:
            saved_data = pickle.load(model_file)

        model = saved_data["model"]
        features = saved_data["feature_columns"]

        if isinstance(input_data, dict):
            input_data = [input_data]

        df = pd.DataFrame(input_data)
        df = pd.get_dummies(df)
        df = df.reindex(columns=features, fill_value=0)

        predictions = model.predict(df)
        return {
            "model_id": model_id,
            "predictions": predictions.tolist(),
        }

    def _get_model_path(self, model_id: str):
        with get_connection() as conn:
            row = conn.execute("SELECT file_path FROM models WHERE id = ?", (model_id,)).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Model not found")
        return row["file_path"]
