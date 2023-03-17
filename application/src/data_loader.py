import os
from abc import ABC, abstractmethod

from numpy import load
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class DataLoader(ABC):
    path: str

    @abstractmethod
    def load_train(self):
        pass

    @abstractmethod
    def load_test(self):
        pass


class BinaryDataLoader(ABC):
    path: str

    def __init__(self, path=os.path.join(os.sep, "data")):
        self.path = path

    def load_train(self):
        x_train = load(os.path.join(self.path, "x_train.npy"), allow_pickle=True)
        y_train = load(os.path.join(self.path, "y_train.npy"), allow_pickle=True)
        return x_train, y_train

    def load_test(self):
        x_test = load(os.path.join(self.path, "x_test.npy"), allow_pickle=True)
        y_test = load(os.path.join(self.path, "y_test.npy"), allow_pickle=True)
        return x_test, y_test

class BuiltInDataLoader(ABC):
    method: function
    
    def __init__(self, method=tf.keras.datasets.cifar10.load_data):
        self.method = method

    def load_train(self):
        (x_train, y_train), (_, _) = self.method()
        return x_train, y_train

    def load_test(self):
        (_, _), (x_test, y_test) = self.method()
        return x_test, y_test