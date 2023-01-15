from typing import List, Dict, Union, Optional

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


class OptimizerConfiguration(BaseModel):
    optimizer: str
    lr: Optional[float]
    momentum: Optional[float]
    weight_decay: Optional[float]


class SchedulerConfiguration(BaseModel):
    scheduler: str
    step_size: Optional[int]
    gamma: Optional[float]


class DPConfiguration(BaseModel):
    num_sampled_clients: int
    init_clip_norm: float = 0.1
    noise_multiplier: float = 1
    server_side_noising: bool = True
    clip_count_stddev: float = None
    clip_norm_target_quantile: float = 0.5
    clip_norm_lr: float = 0.2


class WarmupConfiguration(BaseModel):
    scheduler: str
    warmup_iters: int
    warmup_epochs: int
    warmup_factor: float
    scheduler_conf: SchedulerConfiguration


class HMConfiguration(BaseModel):
    poly_modulus_degree: int = 8192
    coeff_mod_bit_sizes: List[int] = [60, 40, 40]
    scale_bits: int = 40
    scheme: str = "CKKS"


class LOTrainingConfigurationExtended(BaseModel):
    client_type_id: str
    server_address: str
    eval_metrics: List[str]
    eval_func: str
    num_classes: int
    num_rounds: int
    shape: List[int]
    training_id: int
    model_name: str
    model_version: str
    config: List[BasicConfiguration]
    optimizer_config: Optional[OptimizerConfiguration]
    scheduler_config: Optional[SchedulerConfiguration]
    warmup_config: Optional[WarmupConfiguration]
    eval_metrics_value: float

    class Config:
        arbitrary_types_allowed = True


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
    privacy_mechanisms: Dict[str, Union[HMConfiguration, DPConfiguration]] = Field(..., alias='privacy-mechanisms')

    class Config:
        arbitrary_types_allowed = True


class MLModelData(BaseModel):
    meta: Dict[str, str] = Field(None, title="model metadata as key-value pairs")


class MLModel(MLModelData):
    model_name: str = Field(None, title="model identified, str")
    model_version: str = Field(None, title="model version, str")
    model_id: Optional[str] = Field(None,
                                    title="id under which model is stored in gridfs")


class MachineCapabilities(BaseModel):
    storage: Optional[float] = Field(None, title="the amount of storage needed in gigabytes, float")
    RAM: Optional[float] = Field(None, title="the amount of RAM needed in gigabytes, float")
    GPU: bool = Field(False, title="whether the existence of a GPU is needed, bool")
    preinstalled_libraries: Dict[str, str] = Field(None,
                                                   title="a list of necessary/available preinstalled libraries with compliant versions")
    available_models: Dict[str, str] = Field(None,
                                                   title="a list of necessary/available models named by their name and version")