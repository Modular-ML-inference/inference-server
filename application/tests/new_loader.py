from typing import Callable
from application.src.data_loader import DataLoader
import tensorflow





class TransformationBuiltInDataLoader(DataLoader):
    method: Callable
    
    def __init__(self, method=tensorflow.keras.datasets.cifar10.load_data):
        self.method = method

    def load_train(self):
        (x_train, y_train), (_, _) = self.method()
        y_train = tensorflow.keras.utils.to_categorical(y_train, 10)
        return x_train, y_train

    def load_test(self):
        (_, _), (x_test, y_test) = self.method()
        y_test = tensorflow.keras.utils.to_categorical(y_test, 10)
        return x_test, y_test