from abc import ABC, abstractmethod


class ModelLoader(ABC):

    @abstractmethod
    def load(self, model_name, model_version) -> None:
        pass

    @abstractmethod
    def cleanup(self) -> None:
        pass

class TransformationLoader(ABC):

    @abstractmethod
    def load_transformation(self, id):
        pass

    @abstractmethod
    def load_from_config(self):
        pass

class FormatLoader(ABC):

    @abstractmethod
    def load_format(self):
        pass

    @abstractmethod
    def save_format(self):
        pass