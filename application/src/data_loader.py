import os
from abc import ABC, abstractmethod

from numpy import load


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
        x_train = load(os.path.join(self.path, "x_train.npy"))
        y_train = load(os.path.join(self.path, "y_train.npy"))
        return x_train, y_train

    def load_test(self):
        x_test = load(os.path.join(self.path, "x_test.npy"))
        y_test = load(os.path.join(self.path, "y_test.npy"))
        return x_test, y_test
