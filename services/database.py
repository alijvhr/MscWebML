import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATASETS_DIR = DATA_DIR / "datasets"
MODELS_DIR = DATA_DIR / "models"
DB_PATH = DATA_DIR / "ml_api.sqlite3"


def init_db():
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS datasets (
                id TEXT PRIMARY KEY,
                original_name TEXT NOT NULL,
                file_path TEXT NOT NULL,
                rows INTEGER,
                columns INTEGER,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS models (
                id TEXT PRIMARY KEY,
                algorithm TEXT NOT NULL,
                dataset_id TEXT NOT NULL,
                target_column TEXT,
                feature_columns TEXT,
                file_path TEXT NOT NULL,
                score REAL,
                accuracy REAL,
                precision REAL,
                sensitivity REAL,
                specificity REAL,
                f1_score REAL,
                confusion_matrix TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        _add_column_if_missing(cursor, "models", "score", "REAL")
        _add_column_if_missing(cursor, "models", "accuracy", "REAL")
        _add_column_if_missing(cursor, "models", "precision", "REAL")
        _add_column_if_missing(cursor, "models", "sensitivity", "REAL")
        _add_column_if_missing(cursor, "models", "specificity", "REAL")
        _add_column_if_missing(cursor, "models", "f1_score", "REAL")
        _add_column_if_missing(cursor, "models", "confusion_matrix", "TEXT")
        conn.commit()


def _add_column_if_missing(cursor, table_name: str, column_name: str, column_type: str):
    columns = cursor.execute(f"PRAGMA table_info({table_name})").fetchall()
    column_names = [column[1] for column in columns]
    if column_name not in column_names:
        cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")


def get_connection():
    init_db()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn
