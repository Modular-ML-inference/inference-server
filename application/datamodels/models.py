from typing import List, Dict, Optional

from bson import ObjectId
from pydantic import BaseModel, Field


class PyObjectId(ObjectId):

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError('Invalid objectid')
        return ObjectId(v)

    @classmethod
    def __modify_schema__(cls, field_schema):
        field_schema.update(type='string')


class BasicConfiguration(BaseModel):
    config_id: str
    batch_size: int
    steps_per_epoch: int
    epochs: int
    learning_rate: float


class LOTrainingConfiguration(BaseModel):
    client_type_id: str
    server_address: str
    optimizer: str
    eval_metrics: List[str]
    eval_func: str
    num_classes: int
    num_rounds: int
    shape: List[int]
    training_id: int
    model_name: str
    model_version: str
    config: List[BasicConfiguration]
    eval_metrics_value: float

    class Config:
        arbitrary_types_allowed = True


class MLModelData(BaseModel):
    meta: Dict[str, str] = Field(None, title="model metadata as key-value pairs")


class MLModel(MLModelData):
    model_name: str = Field(None, title="model identified, str")
    model_version: str = Field(None, title="model version, str")
    model_id: Optional[str] = Field(None, title="id under which model is stored in gridfs")

