from logging import log, INFO

import keras as keras
import flwr as fl
import requests as requests
import tensorflow as tf

from application.additional.exceptions import BadConfigurationError
from application.additional.utils import ModelLoader
from application.config import ORCHESTRATOR_ADDRESS
from application.src.clientbuilder import FlowerClientBuilder
from application.src.data_loader import BinaryDataLoader

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
    "terminateonnan": tf.keras.callbacks.TerminateOnNan
}


class KerasBuilder(FlowerClientBuilder):

    def __init__(self, training_id, configuration):
        self.configuration = configuration
        self.client = KerasClient(training_id, configuration)

    def product(self):
        self.add_optimizer()
        self.add_model()
        return self.client

    def add_model(self):
        loader = ModelLoader()
        log(INFO, "model")
        loader.load(self.configuration.model_name, self.configuration.model_version)
        self.client.model = keras.models.load_model(ModelLoader.temp_dir)
        loader.cleanup()
        self.client.model.compile(self.client.optimizer, self.configuration.eval_func,
                           metrics=self.configuration.eval_metrics)

    def add_optimizer(self) -> None:
        config = default_twotronics_config.optimizer_config
        input_conf = config.dict(exclude_unset=True)
        input_conf.pop("optimizer")
        try:
            optimizer = keras_optimizers[self.configuration.optimizer](
                **input_conf)
        except AttributeError as e:
            raise BadConfigurationError("optimizer")
        self.client.optimizer = optimizer


class KerasClient(fl.client.NumPyClient):

    def __init__(self, training_id, config):
        self.priv_config = config
        self.round = 1
        self.training_id = training_id
        self.data_loader = BinaryDataLoader()
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
        self.model.fit(self.x_train, self.y_train, epochs=epochs, batch_size=batch_size,
                       steps_per_epoch=steps_per_epoch, callbacks=[MyCustomCallback()])
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
        query = requests.get(f"{ORCHESTRATOR_ADDRESS}/recoverTrainingEpochs"f"/{str(epoch)}"f"/{str(epochs)}")
        return query
