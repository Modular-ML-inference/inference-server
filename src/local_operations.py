import flwr as fl
import tensorflow as tf


def start_client(config):
    client = LocalOperationsClient(config)
    fl.client.start_numpy_client(server_address=config.server_address, client=client)


# Define local client
class LocalOperationsClient(fl.client.NumPyClient):

    def __init__(self, config):
        self.config = config
        if config.model == "cifar10":
            self.model = tf.keras.applications.MobileNetV2((32, 32, 3), classes=10, weights=None)
        self.model.compile(config.optimizer, "sparse_categorical_crossentropy", metrics=["accuracy"])
        (self.x_train, self.y_train), (self.x_test, self.y_test) = tf.keras.datasets.cifar10.load_data()

    def get_parameters(self):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        self.model.fit(self.x_train, self.y_train, epochs=self.config.epochs, batch_size=self.config.batch_size,
                       steps_per_epoch=self.config.steps_per_epoch)
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test)
        return loss, len(self.x_test), {"accuracy": accuracy}
