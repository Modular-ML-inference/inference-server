from application.additional.object_loaders import TrainingFormatLoader, TrainingTransformationLoader
from application.additional.utils import BasicModelLoader
from data_transformation.exceptions import TransformationPipelineConfigurationInsufficientException
from data_transformation.pipeline import BaseTransformationPipeline
import deepdiff


class TrainTransformationManager:
    '''
    Connects with object loaders to load transformations and various parts of the pipeline
    '''

    def __init__(self):
        self.trans_loader = TrainingTransformationLoader()
        self.format_loader = TrainingFormatLoader()

    def check_dict_intersection(self, trans_form, model_form):
        '''
        A small function to check whether transformed format has all the features 
        that model format wants compliant with it.
        (It can also have some addditional fields, that's fine, we care only about these)
        '''
        diff = deepdiff.DeepDiff(model_form, trans_form)
        check = 'dictionary_item_removed' in diff or 'values_changed' in diff
        return not check

    def check_data_format(self, pipeline):
        format = self.format_loader.load_format()
        new_format = pipeline.transform_format(format)
        model_format = BasicModelLoader().load_format()
        return self.check_dict_intersection(new_format, model_format)

    def load_transformation_pipeline(self, train=True):
        '''
        We will also add here some simple checking on whether the data format is sufficient.
        '''
        transformation_list = self.trans_loader.load_from_config(train=train)
        pipeline = BaseTransformationPipeline(transformation_list)
        if not self.check_data_format(pipeline):
            raise TransformationPipelineConfigurationInsufficientException()
        return pipeline
