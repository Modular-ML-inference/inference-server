from typing import Callable
from application.additional.plugin_managers import TrainTransformationManager
from application.src.data_loader import DataLoader
import tensorflow as tf


class BuiltInDataLoader(DataLoader):
    method: Callable

    def __init__(self, method=tf.keras.datasets.cifar10.load_data):
        self.method = method
        self.trans_manager = TrainTransformationManager()

    def load_train(self):
        (x_train, y_train), (_, _) = self.method()
        pipeline = self.trans_manager.load_transformation_pipeline(train=True)
        (x_train, y_train) = pipeline.transform_data((x_train, y_train))
        return x_train, y_train

    def load_test(self):
        (_, _), (x_test, y_test) = self.method()
        pipeline = self.trans_manager.load_transformation_pipeline(train=False)
        (x_test, y_test) = pipeline.transform_data((x_test, y_test))
        return x_test, y_test
