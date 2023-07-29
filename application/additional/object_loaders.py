import json
import os
import sys
import zipfile
import zipimport
import dill
import requests
import shutil
from application.config import REPOSITORY_ADDRESS
from data_transformation.exceptions import TransformationConfigurationInvalidException


class TrainingTransformationLoader:
    temp_dir = "temp"
    test_config_path = os.path.join("application", "configurations", "transformation_pipeline_test.json")
    train_config_path = os.path.join("application", "configurations", "transformation_pipeline_train.json")
    local_files = os.path.join("application", "local_cache", "transformations")
    rep_name = REPOSITORY_ADDRESS

    def __init__(self, rep_name = REPOSITORY_ADDRESS):
        self.rep_name = rep_name

    def load_transformation(self, id):
        trans_path = os.path.join(self.local_files, f'{id}.pkl')
        try:
            if os.path.isfile(trans_path):
                m_path = os.path.join(self.local_files, f'{id}.zip')
                with zipfile.ZipFile(m_path, mode="r") as archive:
                    archive.printdir()
                    z = archive.infolist()
                importer = zipimport.zipimporter(m_path)
                importer.load_module(id)
                sys.path.insert(0, m_path)
                with open(trans_path, 'rb') as f:
                    transformation = dill.load(f)
                    return transformation()
            else:
                try:
                    with requests.get(f"{self.rep_name}/transformation"
                                f"/{id}",stream=True) as r:
                        with open(trans_path, 'wb') as f:
                            shutil.copyfileobj(r.raw, f)
                    with open(trans_path, 'rb') as f:
                        transformation = dill.load(f)
                    return transformation()
                except BaseException as e:
                    raise TransformationConfigurationInvalidException(id)
        except BaseException as e:
            raise TransformationConfigurationInvalidException(id)
        
    def load_from_config(self, train=True):
        """
        Loads the configuration from the predefined location and constructs a list of transformation from that
        """
        con_path = self.train_config_path if train else self.test_config_path
        if os.path.isfile(con_path):
            with open(con_path, 'rb') as f:
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
        
class TrainingFormatLoader:

    config_path = os.path.join("application", "configurations", "format.json")

    def load_format(self):
        '''Load the format of the data that will be obtained by the model'''
        if os.path.isfile(self.config_path):
            with open(self.config_path, 'rb') as f:
                model_data = json.load(f)
                return model_data
            
    def save_format(self, format):
        with open(self.config_path, 'wb') as f:
            json.dump(format, self.config_path)


class TrainingSetupLoader:
    config_path = os.path.join("application", "configurations", "setup.json")
    loader_path = os.path.join("application", "local_cache", "data_loaders")

    #module_path = os.path.join("application", "local_cache", "protocompiled")
    #service_path = os.path.join("application", "local_cache", "services")    
    #inference_path = os.path.join("inference_application", "local_cache", "inferencers")

    def load_setup(self):
        '''Load the setup of the inferencer'''
        if os.path.isfile(self.config_path):
            with open(self.config_path, 'rb') as f:
                setup_data = json.load(f)
                return setup_data
            
    def load_data_loader(self, loader_id):
        '''Load a given data loader'''
        trans_path = os.path.join(self.loader_path, f'{loader_id}.pkl')
        if os.path.isfile(trans_path):
            m_path = os.path.join(self.loader_path, f'{loader_id}.zip')
            with zipfile.ZipFile(m_path, mode="r") as archive:
                archive.printdir()
                z = archive.infolist()
            importer = zipimport.zipimporter(m_path)
            importer.load_module(loader_id)
            sys.path.insert(0, m_path)
            with open(trans_path, 'rb') as f:
                loader = dill.load(f)
                return loader
    '''
    def load_modules(self, module_list):
        """
        We have to first load modules from the pkg files defined in the modules section of the service config in setup.
        Those files should be placed in local_cache in the protocompiled folder.
        Module packages should be named after the modules to load. 
        """
        for module in module_list:
            m_path = os.path.join(self.module_path, f'{module}.zip')
            with zipfile.ZipFile(m_path, mode="r") as archive:
                archive.printdir()
                z = archive.infolist()
            importer = zipimport.zipimporter(m_path)
            importer.load_module(module)
            sys.path.insert(0, m_path)
    
    def load_method(self, method_name):
        """
        We have to then load method from the pkl file defined in the method section of the service config in setup.
        This file should be placed in local_cache in the protocompiled folder.
        """
        method_path = os.path.join(self.module_path, f'{method_name}.pkl')
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
    '''
