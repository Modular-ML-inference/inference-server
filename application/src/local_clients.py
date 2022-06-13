import pickle

import flwr as fl
import gridfs
import tensorflow as tf
from flwr.common import parameters_to_weights
from pymongo import MongoClient
from starlette.concurrency import run_in_threadpool

from application.config import DB_PORT, FEDERATED_PORT, DATABASE_NAME
from application.src.data_loader import BinaryDataLoader

current_jobs = {}


async def start_client(training_id, config):
    client = LOKerasClient(training_id, config)
    await run_in_threadpool(
        lambda: fl.client.start_numpy_client(server_address=f"{config.server_address}:{FEDERATED_PORT}", client=client))
    if training_id in current_jobs and current_jobs[training_id] > 1:
        current_jobs[training_id] -= 1
    else:
        current_jobs.pop(training_id)


# Define local client
class LOKerasClient(fl.client.NumPyClient):

    def __init__(self, training_id, config):
        self.priv_config = config
        self.round = 1
        self.training_id = training_id
        client = MongoClient(DATABASE_NAME, DB_PORT)
        db = client.local
        db_grid = client.repository_grid
        fs = gridfs.GridFS(db_grid)
        if db.models.find_one({"model_name": config.model_name, "model_version": config.model_version}):
            result = db.models.find_one({"model_id": config.model_id, "model_version": config.model_version})
            # add model json configuration to then properly use the model
            self.model = tf.keras.applications.MobileNetV2(config.shape, classes=config.num_classes, weights=None)
            parameters = pickle.loads(fs.get(result['model_id']).read())
            weights = parameters_to_weights(parameters)
            self.model.set_weights(weights)
        else:
            self.model = tf.keras.applications.MobileNetV2(config.shape, classes=config.num_classes, weights=None)
        self.model.compile(config.optimizer, config.eval_func, metrics=config.eval_metrics)
        self.data_loader = BinaryDataLoader()
        (self.x_train, self.y_train) = self.data_loader.load_train()
        (self.x_test, self.y_test) = self.data_loader.load_test()

    def get_parameters(self):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        epochs = config["config"][0].epochs if config else self.priv_config.config[0].epochs
        batch_size = int(config["config"][0]["batch_size"]) if config else self.priv_config.config[0].batch_size
        steps_per_epoch = config["config"][0].steps_per_epoch if config else self.priv_config.config[0].steps_per_epoch
        self.model.fit(self.x_train, self.y_train, epochs=epochs, batch_size=batch_size,
                       steps_per_epoch=steps_per_epoch)
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, metrics = self.model.evaluate(self.x_test, self.y_test)
        if not isinstance(metrics, list):
            metrics = [metrics]
        evaluations = {m: metrics[i] for i, m in enumerate(self.priv_config.eval_metrics)}
        self.round += 1
        return loss, len(self.x_test), evaluations
