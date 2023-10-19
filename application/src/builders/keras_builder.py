from logging import log, INFO

import keras as keras
import requests as requests
import tensorflow as tf

from application.additional.exceptions import BadConfigurationError, ModelNotLoadedProperlyError
from application.additional.object_loaders import TrainingSetupLoader
from application.additional.utils import BasicModelLoader
from application.config import ORCHESTRATOR_SVR_ADDRESS
from application.src.clientbuilder import FlowerClientInferenceBuilder, FlowerClientTrainingBuilder

keras_optimizers = {
    "sgd": tf.keras.optimizers.SGD,
    "rmsprop": tf.keras.optimizers.RMSprop,
    "adam": tf.keras.optimizers.Adam,
    "adadelta": tf.keras.optimizers.Adadelta,
    "adagrad": tf.keras.optimizers.Adagrad,
    "adamax": tf.keras.optimizers.Adamax,
    "nadam": tf.keras.optimizers.Nadam,
    "ftrl": tf.keras.optimizers.Ftrl
}

keras_callbacks = {
    "earlystopping": tf.keras.callbacks.EarlyStopping,
    "reducelronplateau": tf.keras.callbacks.ReduceLROnPlateau,
    "terminateonnan": tf.keras.callbacks.TerminateOnNaN
}


class KerasBuilder(FlowerClientTrainingBuilder, FlowerClientInferenceBuilder):

    def __init__(self, id, configuration):
        self.configuration = configuration
        self.library = "keras"
        self.id = id

    def prepare_training(self):
        # Load the right data loader
        setup_loader = TrainingSetupLoader()
        setup = setup_loader.load_setup()
        loader_id = setup["data_loader"]
        data_loader = setup_loader.load_data_loader(loader_id)()
        # Load the right client
        client_id = setup["client_library"][self.library]["id"]
        self.client = setup_loader.load_client(client_id)(self.id, self.configuration, data_loader)
        self.client.optimizer = self.add_optimizer()
        self.client.lr_scheduler = self.add_scheduler()
        self.client.model = self.add_model()
        self.client.model.compile(self.client.optimizer, self.configuration.eval_func,
                            metrics=self.configuration.eval_metrics)
        return self.client
    
    def prepare_inference(self) -> None:
        return self.add_model()

    def add_model(self, loader_class = BasicModelLoader):
        loader = loader_class()
        log(INFO, "Model in loading")
        try:
            loader.load(self.configuration.model_name, self.configuration.model_version)
            load_path = loader.check_loading_path(loader.temp_dir)
            log(INFO, f'Model loading at path {load_path}')
            model = keras.models.load_model(load_path)
        except BaseException as e:
            log(INFO, "Model not loaded properly")
            raise ModelNotLoadedProperlyError(model_name = self.configuration.model_name, model_version = self.configuration.model_version)
        finally:
            loader.cleanup()
        log(INFO, "Model properly loaded")
        return model

    def add_optimizer(self) -> None:
        config = self.configuration.optimizer_config
        input_conf = config.dict(exclude_unset=True)
        input_conf.pop("optimizer")
        try:
            optimizer = keras_optimizers[self.configuration.optimizer_config.optimizer](
                **input_conf)
        except AttributeError as e:
            raise BadConfigurationError("optimizer")
        except KeyError as e:
            raise BadConfigurationError("optimizer")
        except TypeError as e:
            raise BadConfigurationError("optimizer")
        log(INFO, "Optimizer added")
        return optimizer

    def add_scheduler(self):
        scheduler_conf = self.configuration.scheduler_config
        input_conf = scheduler_conf.dict(exclude_unset=True)
        input_conf.pop("scheduler")
        try:
            lr_scheduler = keras_callbacks[
                scheduler_conf.scheduler](**input_conf)
        except AttributeError as e:
            raise BadConfigurationError("scheduler/callback")
        except KeyError as e:
            raise BadConfigurationError("scheduler/callback")
        except TypeError as e:
            raise BadConfigurationError("scheduler/callback")
        log(INFO, "Scheduler/callback added")
        return lr_scheduler
