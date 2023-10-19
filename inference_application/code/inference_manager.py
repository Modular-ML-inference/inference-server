
from data_transformation.exceptions import TransformationPipelineConfigurationInsufficientException
from data_transformation.pipeline import BaseTransformationPipeline
from prometheus_client import Summary
from inference_application.code.inferencers.tflite_inferencer import TFLiteInferencer
from inference_application.code.utils import InferenceFormatLoader, InferenceModelLoader, InferenceSetupLoader, InferenceTransformationLoader

TFLITE_PREPARATION_TIME = Summary('tflite_inference_pipeline_preparation_seconds', 'Time needed to set up TFLite inferencer')

library_inferencers = {
    "tflite": TFLiteInferencer
}

class InferenceManager:

    def load_data_format(self):
        format = InferenceFormatLoader().load_format()
        return format

    def load_transformation_pipeline(self):
        transformation_list = InferenceTransformationLoader().load_from_config()
        pipeline = BaseTransformationPipeline(transformation_list)
        return pipeline

    def load_inferencer(self):
        loader = InferenceModelLoader()
        model_conf = loader.check_configuration()
        loader.load(model_conf["model_name"], model_conf["model_version"])
        load_path = loader.check_nested_path(loader.temp_dir)
        setup_loader = InferenceSetupLoader()
        setup_conf = setup_loader.load_setup()
        # Load inferencer
        inferencer = setup_loader.load_inferencer(setup_conf["inference"]["inferencer"])()
        # If not here, check if any additional available in the mounted cache
        model = inferencer.load_model(load_path)
        loader.cleanup()
        model_data_format = model_conf["input_format"]
        inferencer.prepare(model, model_data_format)
        return inferencer

    def __init__(self):
        self.prepare_inference()

    @TFLITE_PREPARATION_TIME.time()
    def prepare_inference(self):
        # Load up all elements
        data_format = self.load_data_format()
        transformation_pipeline = self.load_transformation_pipeline()
        inferencer = self.load_inferencer()
        # Check if the transformed data format agrees with the necessary model format
        new_format = transformation_pipeline.transform_format(data_format)
        if new_format == inferencer.format():
            # If so, reload without exception
            self.data_format = data_format
            self.transformation_pipeline = transformation_pipeline
            self.inferencer = inferencer
        else:
            raise TransformationPipelineConfigurationInsufficientException()

    
    