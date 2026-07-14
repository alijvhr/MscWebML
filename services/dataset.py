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
