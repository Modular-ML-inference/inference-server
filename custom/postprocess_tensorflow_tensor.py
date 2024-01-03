import tensorflow as tf
from data_transformation.transformation import DataTransformation
from datamodels.models import MachineCapabilities


import numpy as np

from typing import List


class PostprocessTensorflowTensor(DataTransformation):
    import numpy as np
    import tensorflow as tf
    from tensorflow.core.framework.tensor_pb2 import TensorProto
    id = "fall-detection-p2-postprocess-tensorflow-tensor"
    description = """A transformation that takes the output and returns a TensorProto ready to get encapsulated in response"""
    parameter_types = {}
    default_values = {}
    outputs = [TensorProto]
    needs = MachineCapabilities(preinstalled_libraries={"tensorflow": "2.12.0"})

    def set_parameters(self, parameters):
        self.params = parameters

    def get_parameters(self):
        return self.params

    def transform_data(self, data):
        data = tf.make_tensor_proto(values = data)
        return data

    def transform_format(self, format):
        return format