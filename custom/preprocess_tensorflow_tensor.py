import tensorflow as tf
from data_transformation.transformation import DataTransformation
from datamodels.models import MachineCapabilities


import numpy as np

from typing import List


class PreprocessTensorflowTensor(DataTransformation):
    import numpy as np
    import tensorflow as tf

    id = "fall-detection-p2-preprocess-tensorflow-tensor"
    description = """A transformation that takes a TensorProto as defined in Tensorflow Core and reshapes it based on the size data"""
    parameter_types = {}
    default_values = {}
    outputs = [np.ndarray]
    needs = MachineCapabilities(preinstalled_libraries={"tensorflow": "2.12.0"})

    def set_parameters(self, parameters):
        self.params = parameters

    def get_parameters(self):
        return self.params

    def transform_data(self, data):
        data = tf.make_ndarray(data.input["array"])
        return data

    def transform_format(self, format):
        if "tensor" in format["data_types"]:
            format["data_types"]["numerical"] = format["data_types"]["tensor"]
        format["data_types"].pop("tensor")
        return format
