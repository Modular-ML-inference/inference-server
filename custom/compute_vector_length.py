from data_transformation.transformation import DataTransformation
from datamodels.models import MachineCapabilities


import numpy as np


from typing import List


class ComputeVectorLength(DataTransformation):
    import numpy as np

    id = "fall-detection-p2-compute-vector-length"
    description = """A transformation that an input vector and computes its length"""
    parameter_types = {"axis": int}
    default_values = {"axis":1}
    outputs = [np.ndarray, np.ndarray]
    needs = MachineCapabilities(preinstalled_libraries={"numpy": "1.23.5"})

    def set_parameters(self, parameters):
        self.params = parameters

    def get_parameters(self):
        return self.params

    def transform_data(self, data):
        data = np.array(data)
        lengths = np.linalg.norm(data, axis=self.params["axis"])
        return data, lengths

    def transform_format(self, format):
        if "numerical" in format["data_types"]:
            format["data_types"]["numerical"]["vector_length"] = True
        return format
