
import os
from pathlib import Path
import shutil
from zipfile import ZipFile
import pytest
import requests
import copy
import tensorflow
from application.additional.exceptions import BadConfigurationError, ModelNotLoadedProperlyError
from data_transformation.loader import ModelLoader

from datamodels.models import LOTrainingConfiguration
from application.src.builders.keras_builder import KerasBuilder, KerasClient

# Set up the test environment.

# First, define the training configuration used here for tests

keras_test_config=LOTrainingConfiguration.parse_obj({"client_type_id": "local1",
  "server_address": "fltrainingcollectorlocal-trainingmain-svc2",
  "eval_metrics": [
    "precision", "recall"
  ],
  "eval_func": "categorical_crossentropy",
  "num_classes": 10,
  "num_rounds": 15,
  "shape": [
    32, 32, 3
  ],
  "training_id": "10",
  "model_name": "test_model_keras",
  "model_version": "first",
  "config": [
    {"config_id": "min_effort",
   "batch_size": "64",
   "steps_per_epoch": "32",
   "epochs": "1",
   "learning_rate": "0.001"}
  ],
  "optimizer_config": {
    "optimizer": "adam",
    "learning_rate":"0.005",
    "amsgrad":"True"
  },
  "scheduler_config": {
    "scheduler": "reducelronplateau",
    "factor":"0.5",
    "min_delta":"0.0003"
  },
  "privacy-mechanisms":{"dp-adaptive":{"num_sampled_clients":"1"}},
  "eval_metrics_value": "0"})

# Then, add a mock of ModelLoader

class KerasMockModelLoader(ModelLoader):
    temp_dir = os.path.join(os.getcwd(), "tests","test_data", "keras_test_model")

    def load(self, model_name, model_version):
        with ZipFile(f'{self.temp_dir}.zip', 'r') as zipObj:
          # Extract all the contents of zip file in current directory
          zipObj.extractall(f'{self.temp_dir}/..')

    def cleanup(self):
        if os.path.exists(self.temp_dir):
          shutil.rmtree(f'{self.temp_dir}')


class KerasMockModelLoaderBroken(ModelLoader):
    temp_dir = os.path.join(os.getcwd(), "tests","test_data", "keras_test_model")

    def load(self, model_name, model_version):
        raise requests.exceptions.RequestException()

    def cleanup(self):
        if os.path.exists(self.temp_dir):
          shutil.rmtree(f'{self.temp_dir}')


def test_model_load_unable():
  builder = KerasBuilder(training_id=13, configuration=keras_test_config)
  builder.client = KerasClient(12, keras_test_config)
  builder.client.optimizer="adam"
  with pytest.raises(ModelNotLoadedProperlyError) as custom_error:
      builder.client.model = builder.add_model(loader_class=KerasMockModelLoaderBroken)
  assert keras_test_config.model_name in str(custom_error)
  assert keras_test_config.model_version in str(custom_error)
  assert not hasattr(builder.client, "model")

def test_model_load_proper():
  builder = KerasBuilder(training_id=13, configuration=keras_test_config)
  builder.client = KerasClient(12, keras_test_config)
  builder.client.optimizer="adam"
  builder.client.model = builder.add_model(loader_class=KerasMockModelLoader)
  assert len(builder.client.model.layers) == 6
  assert type(builder.client.model.optimizer) == tensorflow.optimizers.Adam

def test_optimizer_load_unable_bad_optimizer_name():
  # Mock a bad optimizer configuration (name)
  bad_configuration = copy.deepcopy(keras_test_config)
  bad_configuration.optimizer_config.optimizer = "bad-keyword"
  builder = KerasBuilder(training_id=13, configuration=bad_configuration)
  builder.client = KerasClient(12, keras_test_config)
  with pytest.raises(BadConfigurationError) as custom_error:
      builder.client.optimizer = builder.add_optimizer()
  assert "optimizer" in str(custom_error)
  assert not hasattr(builder.client, "model")

def test_optimizer_load_unable_bad_optimizer_parameters():
  # Mock a bad optimizer configuration (parameters)
  bad_configuration = copy.deepcopy(keras_test_config)
  bad_configuration.optimizer_config.lambd = 0.0013
  builder = KerasBuilder(training_id=13, configuration=bad_configuration)
  builder.client = KerasClient(12, keras_test_config)
  with pytest.raises(BadConfigurationError) as custom_error:
      builder.client.optimizer = builder.add_optimizer()
  assert "optimizer" in str(custom_error)
  assert not hasattr(builder.client, "optimizer")

def test_optimizer_load_proper():
  builder = KerasBuilder(training_id=13, configuration=keras_test_config)
  builder.client = KerasClient(12, keras_test_config)
  builder.client.optimizer = builder.add_optimizer()
  assert type(builder.client.optimizer) == tensorflow.optimizers.Adam
  assert builder.client.optimizer.learning_rate == 0.005

def test_scheduler_load_unable_bad_scheduler_name():
  # Mock a bad scheduler/callback configuration (name)
  bad_configuration = copy.deepcopy(keras_test_config)
  bad_configuration.scheduler_config.scheduler = "bad-keyword"
  builder = KerasBuilder(training_id=13, configuration=bad_configuration)
  builder.client = KerasClient(12, keras_test_config)
  with pytest.raises(BadConfigurationError) as custom_error:
      builder.client.lr_scheduler = builder.add_scheduler()
  assert "scheduler" in str(custom_error)
  assert not hasattr(builder.client, "scheduler")

def test_scheduler_load_unable_bad_optimizer_parameters():
  # Mock a bad scheduler/callback configuration (parameters)
  bad_configuration = copy.deepcopy(keras_test_config)
  bad_configuration.scheduler_config.monitor = 0.0013
  builder = KerasBuilder(training_id=13, configuration=bad_configuration)
  builder.client = KerasClient(12, keras_test_config)
  with pytest.raises(BadConfigurationError) as custom_error:
      builder.client.lr_scheduler = builder.add_scheduler()
  assert "scheduler" in str(custom_error)
  assert not hasattr(builder.client, "scheduler")

def test_scheduler_load_proper():
  builder = KerasBuilder(training_id=13, configuration=keras_test_config)
  builder.client = KerasClient(12, keras_test_config)
  builder.client.lr_scheduler = builder.add_scheduler()
  assert type(builder.client.lr_scheduler) == tensorflow.keras.callbacks.ReduceLROnPlateau
  assert builder.client.lr_scheduler.min_delta == 0.0003