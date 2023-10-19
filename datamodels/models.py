from typing import List, Dict, Tuple, Union, Optional, Any

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
    # pytorch-specific config (loosely speaking, some concepts repeat)
    optimizer: str
    lr: Optional[float]
    rho: Optional[float]
    eps: Optional[float]
    foreach: Optional[bool]
    maximize: Optional[bool]
    lr_decay: Optional[float]
    betas: Optional[Tuple[float, float]]
    etas: Optional[Tuple[float, float]]
    step_sizes: Optional[Tuple[float, float]]
    lambd: Optional[float]
    alpha: Optional[float]
    t0: Optional[float]
    max_iter: Optional[int]
    max_eval: Optional[int]
    tolerance_grad: Optional[float]
    tolerance_change: Optional[float]
    history_size: Optional[int]
    line_search_fn: Optional[str]
    momentum_decay: Optional[float]
    dampening: Optional[float]
    centered: Optional[bool]
    nesterov: Optional[bool]
    momentum: Optional[float]
    weight_decay: Optional[float]
    # keras specific config
    amsgrad: Optional[bool]
    learning_rate: Optional[float]
    name: Optional[str]
    clipnorm: Optional[float]
    global_clipnorm: Optional[float]
    use_ema: Optional[bool]
    ema_momentum: Optional[float]
    ema_overwrite_frequency: Optional[int]
    jit_compile: Optional[bool]
    epsilon: Optional[float]
    clipvalue: Optional[float]
    initial_accumulator_value: Optional[float]
    beta_1: Optional[float]
    beta_2: Optional[float]
    beta_2_decay: Optional[float]
    epsilon_1: Optional[float]
    epsilon_2: Optional[float]
    learning_rate_power: Optional[float]
    l1_regularization_strength: Optional[float]
    l2_regularization_strength: Optional[float]
    l2_shrinkage_regularization_strength: Optional[float]
    beta: Optional[float]


class SchedulerConfiguration(BaseModel):
    scheduler: str
    # Let's start with PyTorch
    step_size: Optional[int]
    gamma: Optional[float]
    last_epoch: Optional[int]
    verbose: Optional[Union[bool, int]]
    milestones: Optional[List[int]]
    factor: Optional[float]
    total_iters: Optional[int]
    start_factor: Optional[float]
    end_factor: Optional[float]
    # And then go with Keras callbacks.
    monitor: Optional[str]
    min_delta: Optional[float]
    patience: Optional[int]
    mode: Optional[str]
    baseline: Optional[float]
    restore_best_weights: Optional[bool]
    start_from_epoch: Optional[int]
    cooldown: Optional[int]
    min_lr: Optional[float]


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


class TrainingException(BaseModel):
    stage: str
    reason: str


class HMConfiguration(BaseModel):
    poly_modulus_degree: int = 8192
    coeff_mod_bit_sizes: List[int] = [60, 40, 40]
    scale_bits: int = 40
    scheme: str = "CKKS"


class LOTrainingConfiguration(BaseModel):
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
    privacy_mechanisms: Dict[str, Union[HMConfiguration,
                                        DPConfiguration]] = Field(..., alias='privacy-mechanisms')
    eval_metrics_value: Optional[float]

    class Config:
        arbitrary_types_allowed = True


class MLModelData(BaseModel):
    meta: Dict[str, str] = Field(
        None, title="model metadata as key-value pairs")


class MLModel(MLModelData):
    model_name: str = Field(None, title="model identified, str")
    model_version: str = Field(None, title="model version, str")
    model_id: Optional[str] = Field(None,
                                    title="id under which model is stored in gridfs")


class MachineCapabilities(BaseModel):
    storage: Optional[float] = Field(
        None, title="the amount of storage needed in gigabytes, float")
    RAM: Optional[float] = Field(
        None, title="the amount of RAM needed in gigabytes, float")
    GPU: bool = Field(
        False, title="whether the existence of a GPU is needed, bool")
    preinstalled_libraries: Dict[str, str] = Field(None,
                                                   title="a list of necessary/available preinstalled libraries with compliant versions")
    available_models: Dict[str, str] = Field(None,
                                             title="a list of necessary/available models named by their name and version")


class FLDataTransformation(BaseModel):
    id: str
    description: Optional[str] = Field(None,
                                       title="the available data explaining the purpose of a given transformation")
    parameter_types: Dict[str, str] = Field(
        None, title="the list of input parameters and their types")
    default_values: Dict[str, Any] = Field(None,
                                           title="for the parameters having default values, input them along with the description of values")
    outputs: List[str] = Field(
        None, title="List of outputs and their expected types")
    needs: MachineCapabilities


class FLDataTransformationConfig(BaseModel):
    id: str
    params: Dict[str, Any]


class FLDataTransformationPipelineConfig(BaseModel):
    configuration: Dict[str, List[FLDataTransformationConfig]]


class WebsocketResponse(BaseModel):
    status: str
    message: Optional[str]
