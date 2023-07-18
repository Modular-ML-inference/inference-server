from logging import log, INFO

import keras as keras
import flwr as fl
import requests as requests
import tensorflow as tf
import os

from application.additional.exceptions import BadConfigurationError, ModelNotLoadedProperlyError
from application.additional.utils import BasicModelLoader
from application.config import ORCHESTRATOR_SVR_ADDRESS
from application.src.clientbuilder import FlowerClientInferenceBuilder, FlowerClientTrainingBuilder
from application.src.data_loader import BinaryDataLoader, BuiltInDataLoader

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
        self.id = id

    def prepare_training(self):
        self.client = KerasClient(self.id, self.configuration)
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

class KerasClient(fl.client.NumPyClient):

    def __init__(self, training_id, config):
        self.priv_config = config
        self.round = 1
        self.training_id = training_id
        self.data_loader = BuiltInDataLoader()
        (self.x_train, self.y_train) = self.data_loader.load_train()
        (self.x_test, self.y_test) = self.data_loader.load_test()

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        global epochs
        epochs = self.priv_config.config[0].epochs
        batch_size = self.priv_config.config[0].batch_size
        steps_per_epoch = self.priv_config.config[0].steps_per_epoch
        # Set up the necessary callback
        callbacks = [MyCustomCallback()]
        # Add a custom callback if so set up
        if self.lr_scheduler:
            callbacks.append(self.lr_scheduler)
        self.model.fit(self.x_train, self.y_train, epochs=epochs, batch_size=batch_size,
                       steps_per_epoch=steps_per_epoch, callbacks=callbacks)
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        metrics = ["loss"] + [m for m in self.priv_config.eval_metrics]
        a = self.model.evaluate(self.x_test, self.y_test)
        evaluations = {m: a[i] for i, m in enumerate(metrics)}
        loss = evaluations["loss"]
        del evaluations["loss"]
        self.round += 1
        return loss, len(self.x_test), evaluations

class MyCustomCallback(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch , logs=None):
        try:
            query = requests.get(f"{ORCHESTRATOR_SVR_ADDRESS}/recoverTrainingEpochs"f"/{str(epoch)}"f"/{str(epochs)}")
            return query
        except requests.exceptions.ConnectionError as e:
            log(INFO, f'Could not connect to orchestrator on {ORCHESTRATOR_SVR_ADDRESS}')
