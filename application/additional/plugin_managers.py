from application.additional.object_loaders import TrainingTransformationLoader
from data_transformation.pipeline import BaseTransformationPipeline


class TrainTransformationManager:
    '''
    Connects with object loaders to load transformations and various parts of the pipeline
    '''

    def load_data_format(self):
        format = TrainingTransformationLoader().load_format()
        return format

    def load_transformation_pipeline(self, train=True):
        transformation_list = TrainingTransformationLoader().load_from_config(train=train)
        pipeline = BaseTransformationPipeline(transformation_list)
        return pipeline