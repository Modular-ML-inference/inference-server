from logging import INFO, log

import keras as keras
import requests as requests
import tensorflow as tf

from application.additional.exceptions import ModelNotLoadedProperlyError
from application.additional.utils import BasicModelLoader
from application.src.clientbuilder import FlowerClientInferenceBuilder


class TFLiteBuilder(FlowerClientInferenceBuilder):

    def __init__(self, id, configuration):
        self.configuration = configuration
        self.id = id

    def prepare_inference(self):
        interpreter = self.add_interpreter()
        interpreter.allocate_tensors()
        return interpreter

    def add_interpreter(self, loader_class=BasicModelLoader):
        loader = loader_class()
        log(INFO, "Model in loading")
        try:
            loader.load(self.configuration.model_name,
                        self.configuration.model_version)
            load_path = loader.check_loading_path(loader.temp_dir)
            log(INFO, f'Model loading at path {load_path}')
            interpreter = tf.lite.Interpreter(model_path=load_path)
        except BaseException as e:
            log(INFO, "Model not loaded properly")
            raise ModelNotLoadedProperlyError(
                model_name=self.configuration.model_name, model_version=self.configuration.model_version)
        finally:
            loader.cleanup()
        log(INFO, "Model properly loaded")
        return interpreter
