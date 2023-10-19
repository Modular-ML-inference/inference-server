import os
from abc import ABC, abstractmethod

from numpy import load
import tensorflow as tf
from typing import Callable
from torch.utils.data import DataLoader
from application.additional.plugin_managers import TrainTransformationManager

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class DataLoader(ABC):
    path: str

    @abstractmethod
    def load_train(self):
        pass

    @abstractmethod
    def load_test(self):
        pass


class BinaryDataLoader(DataLoader):
    path: str

    def __init__(self, path=os.path.join(os.sep, "data")):
        self.path = path

    def load_train(self):
        x_train = load(os.path.join(
            self.path, "x_train.npy"), allow_pickle=True)
        y_train = load(os.path.join(
            self.path, "y_train.npy"), allow_pickle=True)
        return x_train, y_train

    def load_test(self):
        x_test = load(os.path.join(self.path, "x_test.npy"), allow_pickle=True)
        y_test = load(os.path.join(self.path, "y_test.npy"), allow_pickle=True)
        return x_test, y_test


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
