from abc import ABC, abstractmethod


class FLDatasetManager(ABC):

    def __init__(self, data_format, data_folder="/data", preprocessed_folder="/preprocessed"):
        self.data_format = data_format
        self.data_folder = data_folder
        self.preprocessed_folder = preprocessed_folder
        self.dataset = self.load_dataset()

    @abstractmethod
    def load_dataset(self):
        """Load the data as a sort of object"""

    @abstractmethod
    def set_dataset_transformation_pipeline(self, pipeline):
        """Set the transformation pipeline that will be used statically for the dataset"""

    @abstractmethod
    def get_dataset_transformation_pipeline(self, pipeline):
        """Get the transformation pipeline that will be used staticallly for the dataset"""

    @abstractmethod
    def transform_dataset(self):
        """Use the transformation pipeline to change the format of the dataset"""

