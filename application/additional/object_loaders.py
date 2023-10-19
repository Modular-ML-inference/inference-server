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
    test_config_path = os.path.join(
        "application", "configurations", "transformation_pipeline_test.json")
    train_config_path = os.path.join(
        "application", "configurations", "transformation_pipeline_train.json")
    local_files = os.path.join("application", "local_cache", "transformations")
    rep_name = REPOSITORY_ADDRESS

    def __init__(self, rep_name=REPOSITORY_ADDRESS):
        self.rep_name = rep_name

    def load_transformation(self, id):
        trans_path = os.path.join(self.local_files, f'{id}.pkl')
        ephem_path = f'{self.temp_dir}.zip'
        try:
            if not os.path.isfile(trans_path):
                # if the right file is not here, try to download it
                try:
                    with requests.get(f"{self.rep_name}/transformation"
                                      f"/{id}", stream=True) as r:
                        with open(ephem_path, 'wb') as f:
                            shutil.copyfileobj(r.raw, f)
                    # First we extract all the downloaded files to the cache dir
                    with zipfile.ZipFile(ephem_path, 'r') as zipObj:
                        # Extract all the contents of zip file in current transformations directory
                        zipObj.extractall(self.local_files)
                    # now, since we have extracted everything to the transformations
                    # we have to cleanup
                    self.cleanup()
                except BaseException as e:
                    raise TransformationConfigurationInvalidException(id)
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
        except BaseException as e:
            raise TransformationConfigurationInvalidException(id)

    def cleanup(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(f'{self.temp_dir}')
        if os.path.exists(f'{self.temp_dir}.zip'):
            os.remove(f'{self.temp_dir}.zip')

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
    client_path = os.path.join('application', 'local_cache', 'clients')

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

    def load_client(self, client_id):
        '''Load a given client'''
        trans_path = os.path.join(self.client_path, f'{client_id}.pkl')
        if os.path.isfile(trans_path):
            m_path = os.path.join(self.client_path, f'{client_id}.zip')
            with zipfile.ZipFile(m_path, mode="r") as archive:
                archive.printdir()
                z = archive.infolist()
            importer = zipimport.zipimporter(m_path)
            importer.load_module(client_id)
            sys.path.insert(0, m_path)
            with open(trans_path, 'rb') as f:
                client = dill.load(f)
                return client
