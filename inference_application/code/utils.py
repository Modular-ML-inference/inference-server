

import json
from logging import INFO, log
import os
import shutil
from zipfile import ZipFile
import requests
import numpy as np
from pickle import load
import dill
from data_transformation.exceptions import TransformationConfigurationInvalidException
from data_transformation.loader import ModelLoader
from inference_application.config import REPOSITORY_ADDRESS


class InferenceModelLoader(ModelLoader):

    temp_dir = "temp"
    config_path = os.path.join("inference_application", "configurations", "model.json")
    local_files = os.path.join("inference_application", "local_cache", "models")

    def __init__(self, rep_name = REPOSITORY_ADDRESS):
        self.rep_name = rep_name
    
    def check_library(self, model_name, model_version):
        if os.path.isfile(self.config_path):
            # TODO: probably add more jsonschema specific handling later
            # along with better error handling
            with open(self.config_path, 'rb') as f:
                model_data = json.load(f)
                return model_data["library"]
        else:
            # If not in local files, check repository
            with requests.get(f"{self.rep_name}/model/meta",
                            params={"model_name": model_name,
                                    "model_version": model_version}) \
                    as r:
                return r.json()["meta"]["library"]
            
    def check_nested_path(self, temp):
        '''Checks how nested was the zipped file in order to load it correctly'''
        nested_files = os.listdir(temp)
        log(INFO, f'In model directory there are following files {nested_files}')
        if len(nested_files)==1 and os.path.isdir(os.path.join(temp, nested_files[0])):
            return self.check_nested_path(os.path.join(temp, nested_files[0]))
        else:
            return temp
        
    def check_configuration(self):
        if os.path.isfile(self.config_path):
            with open(self.config_path, 'rb') as f:
                model_data = json.load(f)
                return model_data
        
    def load_format(self, model_name, model_version):
        '''Load the data format accepted by the model'''
        # TODO: add later some more dynamic format loading
        local_file_path = os.path.join(local_file_path, model_name, f'{model_version}.zip')
        if os.path.isfile(local_file_path) and os.path.isfile(self.config_path):
            with open(self.config_path, 'rb') as f:
                model_data = json.load(f)
                return model_data["input_format"]

    def load(self, model_name, model_version):
        '''Properly load the model from files'''
        local_file_path = os.path.join(self.local_files, model_name, f'{model_version}.zip')
        if os.path.isfile(local_file_path) and os.path.isfile(self.config_path):
            with ZipFile(local_file_path, 'r') as zipObj:
                # Extract all the contents of zip file in current directory
                zipObj.extractall(f'{self.temp_dir}')
        else:
            with requests.get(f"{self.rep_name}/model"
                              f"/{model_name}/{model_version}",
                              stream=True) as r:
                with open(f'{self.temp_dir}.zip', 'wb') as f:
                    shutil.copyfileobj(r.raw, f)
            with ZipFile(f'{self.temp_dir}.zip', 'r') as zipObj:
                # Extract all the contents of zip file in current directory
                zipObj.extractall(f'{self.temp_dir}')

    def cleanup(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(f'{self.temp_dir}')
        if os.path.exists(f'{self.temp_dir}.zip'):
            os.remove(f'{self.temp_dir}.zip')

class InferenceTransformationLoader:

    config_path = os.path.join("inference_application", "configurations", "transformation_pipeline.json")
    local_files = os.path.join("inference_application", "local_cache", "transformations")

    def __init__(self, rep_name = REPOSITORY_ADDRESS):
        self.rep_name = rep_name

    def load_transformation(self, id):
        trans_path = os.path.join(self.local_files, f'{id}.pkl')
        try:
            if os.path.isfile(trans_path):
                with open(trans_path, 'rb') as f:
                    transformation = dill.load(f)
                    return transformation()
            else:
                raise TransformationConfigurationInvalidException(id)
        except BaseException as e:
            raise TransformationConfigurationInvalidException(id)
        
    def load_from_config(self):
        """
        Loads the configuration from the predefined location and constructs a list of transformation from that
        """
        if os.path.isfile(self.config_path):
            with open(self.config_path, 'rb') as f:
                config = json.load(f)
            pipeline = []
            for transform in config:
                trans_class = self.load_transformation(transform["id"])
                if transform["parameters"]:
                    trans_class.set_parameters(transform["parameters"])
                pipeline.append(trans_class)
            return pipeline
        else:
            return []
        
class InferenceFormatLoader:

    config_path = os.path.join("inference_application", "configurations", "format.json")

    def load_format(self):
        '''Load the format of the data that will be obtained by the model'''
        if os.path.isfile(self.config_path):
            with open(self.config_path, 'rb') as f:
                model_data = json.load(f)
                return model_data
            
    def save_format(self, format):
        with open(self.config_path, 'wb') as f:
            json.dump(format, self.config_path)

def reconstruct_shape(data, shape):
    """
    A method that reconstructs the original array sent through grpc by properly employing shape
    """
    data = np.array(data).reshape(tuple(shape))
    return data

def deconstruct_shape(inference):
    """
    In order to send the data in a format compliant with grpc, we have to flattten the array to a list
    and store separately information about the shape.
    """
    inference_array = np.array(inference)
    shape = list(inference_array.shape)
    inference = inference_array.flatten()
    return inference, shape