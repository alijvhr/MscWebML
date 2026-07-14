from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from services.database import init_db
from services.dataset import DatasetService
from services.encode import EncodeService
from services.predict import PredictService
from services.preprocess import PreprocessService
from services.train import TrainService

app = FastAPI(title="Student Machine Learning API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

dataset_service = DatasetService()
preprocess_service = PreprocessService()
encode_service = EncodeService()
train_service = TrainService()
predict_service = PredictService()


class PreprocessRequest(BaseModel):
    dataset_id: str
    fill_missing: bool = True
    normalize: bool = True


class EncodeRequest(BaseModel):
    dataset_id: str
    method: str = "onehot"
    target_column: str | None = None


class TrainRequest(BaseModel):
    dataset_id: str
    algorithm: str
    target_column: str | None = None
    feature_columns: list[str] | None = None


class PredictRequest(BaseModel):
    model_id: str
    input_data: dict[str, Any] | list[dict[str, Any]]


class UciDatasetRequest(BaseModel):
    uci_id: int | None = None
    name: str | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield

# region Home
@app.get("/")
def root():
    return {"message": "Machine Learning API is running"}


# endregion


# region Dataset Routes
@app.post("/datasets/upload")
async def upload_dataset(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
    return await dataset_service.upload(file)


@app.get("/datasets")
def list_datasets():
    return dataset_service.list_all()


@app.post("/datasets/uci")
def import_uci_dataset(request: UciDatasetRequest):
    return dataset_service.import_uci_dataset(request.uci_id, request.name)


@app.get("/datasets/{dataset_id}")
def get_dataset(dataset_id: str, rows: int = 5):
    return dataset_service.preview(dataset_id, rows)


# endregion


# region Preprocess Routes
@app.post("/preprocess")
def preprocess_dataset(request: PreprocessRequest):
    return preprocess_service.preprocess(
        request.dataset_id,
        fill_missing=request.fill_missing,
        normalize=request.normalize,
    )


# endregion


# region Encode Routes
@app.post("/encode")
def encode_dataset(request: EncodeRequest):
    return encode_service.encode(request.dataset_id, request.method, request.target_column)


# endregion


# region Train Routes
@app.post("/train")
def train_model(request: TrainRequest):
    return train_service.train(
        dataset_id=request.dataset_id,
        algorithm=request.algorithm,
        target_column=request.target_column,
        feature_columns=request.feature_columns,
    )


@app.get("/models")
def list_models():
    return train_service.list_all()


@app.get("/models/{model_id}")
def get_model(model_id: str):
    return train_service.get_by_id(model_id)


# endregion


# region Predict Routes
@app.post("/predict")
def predict(request: PredictRequest):
    return predict_service.predict(request.model_id, request.input_data)


# endregion
