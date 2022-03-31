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
    shape: List[int]
    model_id: int
    model_version: int
    config: List[BasicConfiguration]

    class Config:
        arbitrary_types_allowed = True


class MLModelData(BaseModel):
    meta: Dict[str, str] = Field(None, title="model metadata as key-value pairs")


class MLModel(MLModelData):
    id: int = Field(None, title="model identified, numeric")
    version: int = Field(None, title="model version, numeric")
    model_id: Optional[str] = Field(None, title="id under which model is stored in gridfs")

