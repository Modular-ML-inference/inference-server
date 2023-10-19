
from logging import log, INFO

import keras as keras
import flwr as fl
import requests as requests
import os


class PickledKerasClient(fl.client.NumPyClient):

    def __init__(self, training_id, config, data_loader):
        self.priv_config = config
        self.round = 1
        self.training_id = training_id
        self.data_loader = data_loader
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
    def on_epoch_begin(self, epoch, logs=None):
        try:
            ORCHESTRATOR_SVR_ADDRESS = os.environ['ORCHESTRATOR_SVR_ADDRESS']
            query = requests.get(
                f"{ORCHESTRATOR_SVR_ADDRESS}/recoverTrainingEpochs"f"/{str(epoch)}"f"/{str(epochs)}")
            return query
        except requests.exceptions.ConnectionError as e:
            log(INFO,
                f'Could not connect to orchestrator on {ORCHESTRATOR_SVR_ADDRESS}')
