

import json
from logging import INFO, log
import os
import shutil
import sys
from zipfile import ZipFile
import zipfile
import zipimport
import requests
import numpy as np
import dill
from data_transformation.exceptions import TransformationConfigurationInvalidException
from data_transformation.loader import ModelLoader
from inference_application.config import if_env

TIMEOUT = 600

class InferenceModelLoader(ModelLoader):
    '''
    Loader class responsible for models. Loads them from the Component Repository and from the local_cache directory
    '''

    temp_dir = "temp"
    config_path = os.path.join(
        "inference_application", "configurations", "model.json")
    local_files = os.path.join(
        "inference_application", "local_cache", "models")
    rep_name = if_env('REPOSITORY_ADDRESS')

    def __init__(self, rep_name=if_env('REPOSITORY_ADDRESS')):
        self.rep_name = rep_name

    def check_library(self, model_name, model_version):
        '''
        Check which library is required to open the model.
        '''
        if os.path.isfile(self.config_path):
            with open(self.config_path, 'rb') as f:
                model_data = json.load(f)
                return model_data["library"]
        else:
            # If not in local files, check repository
            with requests.get(f"{self.rep_name}/model/meta",
                              params={"model_name": model_name,
                                      "model_version": model_version}, timeout=TIMEOUT) \
                    as r:
                return r.json()["meta"]["library"]

    def check_nested_path(self, temp):
        '''Checks how nested was the zipped file in order to load it correctly'''
        nested_files = os.listdir(temp)
        log(INFO,
            f'In model directory there are following files {nested_files}')
        if len(nested_files) == 1 and os.path.isdir(os.path.join(temp, nested_files[0])):
            return self.check_nested_path(os.path.join(temp, nested_files[0]))
        else:
            return temp

    def check_configuration(self):
        '''
        Get model configuration
        '''
        if os.path.isfile(self.config_path):
            with open(self.config_path, 'rb') as f:
                model_data = json.load(f)
                return model_data

    def load_format(self, model_name, model_version):
        '''Load the data format accepted by the model'''
        local_file_path = os.path.join(
            self.local_files, model_name, f'{model_version}.zip')
        if os.path.isfile(local_file_path) and os.path.isfile(self.config_path):
            with open(self.config_path, 'rb') as f:
                model_data = json.load(f)
                return model_data["input_format"]

    def load(self, model_name, model_version):
        '''Properly load the model from files'''
        local_file_path = os.path.join(
            self.local_files, model_name, f'{model_version}.zip')
        if os.path.isfile(local_file_path) and os.path.isfile(self.config_path):
            with ZipFile(local_file_path, 'r') as zipObj:
                # Extract all the contents of zip file in current directory
                zipObj.extractall(f'{self.temp_dir}')
        else:
            with requests.get(f"{self.rep_name}/model"
                              f"/{model_name}/{model_version}",
                              stream=True, timeout=TIMEOUT) as r:
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
    '''
    Loader class responsible for data transformations. Loads them from the Component Repository and from the local_cache directory.
    '''
    temp_dir = "temp"
    pre_config_path = os.path.join(
        "inference_application", "configurations", "preprocessing_pipeline.json")
    post_config_path = os.path.join(
        "inference_application", "configurations", "postprocessing_pipeline.json")
    local_files = os.path.join(
        "inference_application", "local_cache", "transformations")
    rep_name = if_env('REPOSITORY_ADDRESS')

    def __init__(self, rep_name=if_env('REPOSITORY_ADDRESS')):
        self.rep_name = rep_name

    def load_transformation(self, trans_id):
        '''
        Load the transformation, including the model in which the transformation is located.
        '''
        trans_path = os.path.join(self.local_files, f'{trans_id}.pkl')
        ephem_path = f'{self.temp_dir}.zip'
        try:
            if not os.path.isfile(trans_path):
                # if the right file is not here, try to download it
                try:
                    with requests.get(f"{self.rep_name}/transformation"
                                      f"/{trans_id}", stream=True, timeout=TIMEOUT) as r:
                        with open(ephem_path, 'wb') as f:
                            shutil.copyfileobj(r.raw, f)
                    # First we extract all the downloaded files to the cache dir
                    with ZipFile(ephem_path, 'r') as zipObj:
                        # Extract all the contents of zip file in current transformations directory
                        zipObj.extractall(self.local_files)
                    # now, since we have extracted everything to the transformations
                    # we have to cleanup
                    self.cleanup()
                except BaseException as e:
                    raise TransformationConfigurationInvalidException(trans_id) from e
            m_path = os.path.join(self.local_files, f'{trans_id}.zip')
            with zipfile.ZipFile(m_path, mode="r") as archive:
                archive.printdir()
                archive.infolist()
            importer = zipimport.zipimporter(m_path)
            importer.exec_module(trans_id)
            sys.path.insert(0, m_path)
            with open(trans_path, 'rb') as f:
                transformation = dill.load(f)
                return transformation()
        except BaseException as e:
            raise TransformationConfigurationInvalidException(trans_id) from e

    def cleanup(self):
        '''
        Clean up after dynamically loading the transformation from the Component Repository
        '''
        if os.path.exists(self.temp_dir):
            shutil.rmtree(f'{self.temp_dir}')
        if os.path.exists(f'{self.temp_dir}.zip'):
            os.remove(f'{self.temp_dir}.zip')

    def load_from_config(self, preprocessing=True):
        """
        Loads the configuration from the predefined location and constructs a list of transformation from that
        """
        config_path = self.pre_config_path if preprocessing else self.post_config_path
        if os.path.isfile(config_path):
            with open(config_path, 'rb') as f:
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
    '''
    An object responsible for loading the format.
    '''

    config_path = os.path.join(
        "inference_application", "configurations", "format.json")

    def load_format(self):
        '''Load the format of the data that will be obtained by the model'''
        if os.path.isfile(self.config_path):
            with open(self.config_path, 'rb') as f:
                model_data = json.load(f)
                return model_data

    def save_format(self, data_format):
        '''Save the current format as the input format'''
        with open(self.config_path, 'wb'):
            json.dump(data_format, self.config_path)


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


class InferenceSetupLoader:
    '''
    A loader responsible for the general setup, that is, the server, the inferencers and other extra modules.
    '''

    temp_dir = "temp"
    config_path = os.path.join(
        "inference_application", "configurations", "setup.json")
    service_path = os.path.join(
        "inference_application", "local_cache", "services")
    inference_path = os.path.join(
        "inference_application", "local_cache", "inferencers")
    rep_name = if_env('REPOSITORY_ADDRESS')

    def load_setup(self):
        '''Load the setup of the inferencer'''
        if os.path.isfile(self.config_path):
            with open(self.config_path, 'rb') as f:
                setup_data = json.load(f)
                return setup_data
            
    def check_service_availability(self, service_id):
        '''
        Check, whether the servicer has been already downloaded locally.
        '''
        serv_path = os.path.join(self.service_path, f'{service_id}.pkl')
        ephem_path = f'{self.temp_dir}.zip'
        if not os.path.isfile(serv_path):
            # if the right file is not here, try to download it
            try:
                with requests.get(f"{self.rep_name}/service"
                                      f"/{service_id}", stream=True, timeout=TIMEOUT) as r:
                    with open(ephem_path, 'wb') as f:
                        shutil.copyfileobj(r.raw, f)
                    # First we extract all the downloaded files to the cache dir
                with ZipFile(ephem_path, 'r') as zipObj:
                        # Extract all the contents of zip file in current service directory
                    zipObj.extractall(self.service_path)
                    # now, since we have extracted everything to the services
                    # we have to cleanup
                self.cleanup()
            except BaseException as e:
                raise TransformationConfigurationInvalidException(service_id) from e
                
    def check_inferencer_availability(self, inferencer_id):
        '''
        Check, whether the inferencer has been already downloaded locally.
        '''
        inf_path = os.path.join(self.inference_path, f'{inferencer_id}.pkl')
        ephem_path = f'{self.temp_dir}.zip'
        if not os.path.isfile(inf_path):
            # if the right file is not here, try to download it
            try:
                with requests.get(f"{self.rep_name}/inferencer"
                                      f"/{inferencer_id}", stream=True, timeout=TIMEOUT) as r:
                    with open(ephem_path, 'wb') as f:
                            shutil.copyfileobj(r.raw, f)
                # First we extract all the downloaded files to the cache dir
                with ZipFile(ephem_path, 'r') as zip_obj:
                    # Extract all the contents of zip file in current inferencers directory
                    zip_obj.extractall(inf_path)
                # now, since we have extracted everything to the inferencers
                # we have to cleanup
                self.cleanup()
            except BaseException as exc:
                raise TransformationConfigurationInvalidException(inferencer_id) from exc
                
    def cleanup(self):
        '''
        A function that cleans up after the dynamic download from Component Repository.
        '''
        if os.path.exists(self.temp_dir):
            shutil.rmtree(f'{self.temp_dir}')
        if os.path.exists(f'{self.temp_dir}.zip'):
            os.remove(f'{self.temp_dir}.zip')

    def load_modules(self, module_list):
        """
        We have to first load modules from the pkg files defined in the modules section of the service config in setup.
        Those files should be placed in local_cache in the protocompiled folder.
        Module packages should be named after the modules to load. 
        """
        for module in module_list:
            m_path = os.path.join(self.service_path, f'{module}.zip')
            with zipfile.ZipFile(m_path, mode="r") as archive:
                archive.printdir()
                archive.infolist()
            importer = zipimport.zipimporter(m_path)
            importer.exec_module(module)
            sys.path.insert(0, m_path)

    def load_method(self, method_name):
        """
        We have to then load method from the pkl file defined in the method section of the service config in setup.
        This file should be placed in local_cache in the protocompiled folder.
        """
        method_path = os.path.join(self.service_path, f'{method_name}.pkl')
        with open(method_path, 'rb') as f:
            method = dill.load(f)
        return method

    def load_servicer(self, servicer_name):
        """
        Finally, we have to load service from the pkl file defined in the servicer section of the service config in setup.
        This file should be placed in local_cache in the services folder.
        """
        svc_path = os.path.join(self.service_path, f'{servicer_name}.pkl')
        with open(svc_path, 'rb') as f:
            service = dill.load(f)
        return service

    def load_inferencer(self, inferencer):
        '''Load the selected inferencer'''
        # First module
        m_path = os.path.join(self.inference_path, f'{inferencer}.zip')
        with zipfile.ZipFile(m_path, mode="r") as archive:
            archive.printdir()
        importer = zipimport.zipimporter(m_path)
        importer.exec_module(inferencer)
        sys.path.insert(0, m_path)
        # Then pickled object
        o_path = os.path.join(self.inference_path, f'{inferencer}.pkl')
        with open(o_path, 'rb') as f:
            inferencer = dill.load(f)
        return inferencer
