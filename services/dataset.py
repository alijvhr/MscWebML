import hashlib
from datetime import datetime

import pandas as pd
from fastapi import HTTPException, UploadFile

from services.database import DATASETS_DIR, get_connection


class DatasetService:
    async def upload(self, file: UploadFile):
        content = await file.read()
        dataset_id = hashlib.sha256(content).hexdigest()[:16]
        file_path = DATASETS_DIR / f"{dataset_id}.csv"
        file_path.write_bytes(content)

        try:
            df = pd.read_csv(file_path)
        except Exception as error:
            file_path.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail=f"Cannot read CSV file: {error}")

        with get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO datasets
                (id, original_name, file_path, rows, columns, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    dataset_id,
                    file.filename,
                    str(file_path),
                    len(df),
                    len(df.columns),
                    datetime.utcnow().isoformat(),
                ),
            )
            conn.commit()

        return {
            "dataset_id": dataset_id,
            "original_name": file.filename,
            "rows": len(df),
            "columns": list(df.columns),
        }

    def list_all(self):
        with get_connection() as conn:
            rows = conn.execute(
                "SELECT id, original_name, rows, columns, created_at FROM datasets ORDER BY created_at DESC"
            ).fetchall()
        return {"datasets": [dict(row) for row in rows]}

    def get_path(self, dataset_id: str):
        with get_connection() as conn:
            row = conn.execute("SELECT file_path FROM datasets WHERE id = ?", (dataset_id,)).fetchone()
        if row is None:
            raise HTTPException(status_code=404, detail="Dataset not found")
        return row["file_path"]

    def preview(self, dataset_id: str, rows: int = 5):
        file_path = self.get_path(dataset_id)
        df = pd.read_csv(file_path)
        return {
            "dataset_id": dataset_id,
            "columns": list(df.columns),
            "preview": df.head(rows).fillna("").to_dict(orient="records"),
        }

    def import_uci_dataset(self, uci_id: int | None = None, name: str | None = None):
        if uci_id is None and not name:
            raise HTTPException(status_code=400, detail="Provide a UCI dataset ID or name")

        try:
            from ucimlrepo import fetch_ucirepo
        except ImportError as error:
            raise HTTPException(
                status_code=500,
                detail="ucimlrepo is not installed. Run: pip install ucimlrepo",
            ) from error

        try:
            dataset = fetch_ucirepo(id=uci_id) if uci_id is not None else fetch_ucirepo(name=name)
        except Exception as error:
            raise HTTPException(status_code=400, detail=f"Cannot fetch UCI dataset: {error}") from error

        df = self._uci_to_dataframe(dataset)
        dataset_name = self._uci_dataset_name(dataset, uci_id, name)
        dataset_id = self.save_new_dataset(df, f"uci_{dataset_name}.csv")

        return {
            "message": "UCI dataset imported",
            "dataset_id": dataset_id,
            "original_name": f"uci_{dataset_name}.csv",
            "rows": len(df),
            "columns": list(df.columns),
        }

    def save_new_dataset(self, df: pd.DataFrame, name: str):
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        dataset_id = hashlib.sha256(csv_bytes).hexdigest()[:16]
        file_path = DATASETS_DIR / f"{dataset_id}.csv"
        file_path.write_bytes(csv_bytes)

        with get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO datasets
                (id, original_name, file_path, rows, columns, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (dataset_id, name, str(file_path), len(df), len(df.columns), datetime.utcnow().isoformat()),
            )
            conn.commit()

        return dataset_id

    def _uci_to_dataframe(self, dataset) -> pd.DataFrame:
        data = dataset.data
        features = getattr(data, "features", None)
        targets = getattr(data, "targets", None)
        original = getattr(data, "original", None)

        frames = []
        if features is not None and not features.empty:
            frames.append(features.reset_index(drop=True))

        if targets is not None and not targets.empty:
            target_df = targets.reset_index(drop=True).copy()
            feature_columns = set(frames[0].columns) if frames else set()
            target_df.columns = [
                f"target_{column}" if column in feature_columns else column
                for column in target_df.columns
            ]
            frames.append(target_df)

        if not frames and original is not None and not original.empty:
            frames.append(original.reset_index(drop=True))

        if not frames:
            raise HTTPException(status_code=400, detail="UCI dataset did not include tabular data")

        return pd.concat(frames, axis=1)

    def _uci_dataset_name(self, dataset, uci_id: int | None, requested_name: str | None) -> str:
        metadata = getattr(dataset, "metadata", {}) or {}
        dataset_name = getattr(metadata, "name", None)
        if dataset_name is None and isinstance(metadata, dict):
            dataset_name = metadata.get("name")
        if dataset_name is None:
            dataset_name = requested_name or f"dataset_{uci_id}"
        return "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in dataset_name).strip("_")
