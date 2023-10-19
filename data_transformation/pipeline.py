from abc import ABC, abstractmethod

class DataTransformationPipeline(ABC):
    """
    A class used to construct and present a transformation pipeline.
    Based on a given pipeline schema, it will allow for format and data transformation.
    """

    @abstractmethod
    def construct_pipeline(self, transform_list):
        pass

    @abstractmethod
    def transform_data(self, data):
        pass

    @abstractmethod
    def transform_format(self, format):
        pass


class BaseTransformationPipeline(DataTransformationPipeline):

    def __init__(self, transform_list):
        self.construct_pipeline(transform_list)

    def construct_pipeline(self, transform_list):
        self.pipeline = transform_list

    def transform_data(self, data):
        for transform in self.pipeline:
            data = transform.transform_data(data)
        return data

    def transform_format(self, format):
        for transform in self.pipeline:
            format = transform.transform_format(format)
        return format
