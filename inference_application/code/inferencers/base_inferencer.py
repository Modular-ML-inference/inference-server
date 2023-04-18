from abc import ABC, abstractmethod


class BaseInferencer(ABC):
    @property
    @abstractmethod
    def format(self):
        return self.format

    @abstractmethod
    def prepare(self, inferencer, format):
        pass

    @abstractmethod
    def predict(self, data):
        pass
