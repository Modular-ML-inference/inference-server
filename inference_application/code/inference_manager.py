from data_transformation.exceptions import TransformationPipelineConfigurationInsufficientException
from data_transformation.pipeline import BaseTransformationPipeline
from prometheus_client import Summary
from inference_application.code.inferencers.tflite_inferencer import TFLiteInferencer
from inference_application.code.utils import InferenceFormatLoader, InferenceModelLoader, InferenceSetupLoader, InferenceTransformationLoader

PREPARATION_TIME = Summary('inference_pipeline_preparation_seconds', 'Time needed to set up the inferencer')

library_inferencers = {
    "tflite": TFLiteInferencer
}

class InferenceManager:

    def load_data_format(self):
        data_format = InferenceFormatLoader().load_format()
        return data_format

    def load_preprocessing_pipeline(self):
        transformation_list = InferenceTransformationLoader().load_from_config()
        pipeline = BaseTransformationPipeline(transformation_list)
        return pipeline
    
    def load_postprocessing_pipeline(self):
        transformation_list = InferenceTransformationLoader().load_from_config(preprocessing=False)
        pipeline = BaseTransformationPipeline(transformation_list)
        return pipeline

    def load_inferencer(self):
        loader = InferenceModelLoader()
        model_conf = loader.check_configuration()
        loader.load(model_conf["model_name"], model_conf["model_version"])
        load_path = loader.check_nested_path(loader.temp_dir)
        setup_loader = InferenceSetupLoader()
        setup_conf = setup_loader.load_setup()
        # Check inferenceravailability
        setup_loader.check_inferencer_availability(setup_conf["inference"]["inferencer"])
        # Load inferencer
        inferencer = setup_loader.load_inferencer(setup_conf["inference"]["inferencer"])()
        # Check if GPU should be used for inference
        if "use_cuda" in setup_conf["inference"]:
            device = setup_conf["inference"]["use_cuda"]
            # If not here, check if any additional available in the mounted cache
            model = inferencer.load_model(load_path, use_cuda=device)
        else:
            model = inferencer.load_model(load_path)
        loader.cleanup()
        model_data_format = model_conf["input_format"]
        inferencer.prepare(model, model_data_format)
        return inferencer

    def __init__(self):
        self.prepare_inference()

    @PREPARATION_TIME.time()
    def prepare_inference(self):
        # Load up all elements
        data_format = self.load_data_format()
        preprocessing_pipeline = self.load_preprocessing_pipeline()
        postprocessing_pipeline = self.load_postprocessing_pipeline()
        inferencer = self.load_inferencer()
        # Check if the transformed data format agrees with the necessary model format
        new_format = preprocessing_pipeline.transform_format(data_format)
        if new_format == inferencer.format():
            # If so, reload without exception
            self.data_format = data_format
            self.preprocessing_pipeline = preprocessing_pipeline
            self.postprocessing_pipeline = postprocessing_pipeline
            self.inferencer = inferencer
        else:
            raise TransformationPipelineConfigurationInsufficientException()

    
    