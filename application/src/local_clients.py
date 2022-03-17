import pickle

import flwr as fl
import gridfs
import tensorflow as tf
from pymongo import MongoClient

from application.config import DB_PORT


async def start_client(id, config):
    client = LOKerasClient(config)
    fl.client.start_numpy_client(server_address=f"{config.server_address}:{8080+int(id)}", client=client)


# Define local client
class LOKerasClient(fl.client.NumPyClient):

    def __init__(self, config):
        self.priv_config = config
        client = MongoClient('db', DB_PORT)
        db = client.local
        db_grid = client.repository_grid
        fs = gridfs.GridFS(db_grid)
        if db.models.find_one({"id" : config.id, "version": config.version}):
            result = db.models.find_one({"id": config.id, "version": config.version})
            self.model = pickle.loads(fs.get(result['model_id']).read())
            self.model.__init__(config.shape, classes=config.num_classes, weights=None)
        else:
            self.model = tf.keras.applications.MobileNetV2(config.shape, classes=config.num_classes, weights=None)
        self.model.compile(config.optimizer, config.eval_func, metrics=config.eval_metrics)
        (self.x_train, self.y_train), (self.x_test, self.y_test) = tf.keras.datasets.cifar10.load_data()

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
        return loss, len(self.x_test), evaluations
