from data_transformation.transformation import DataTransformation
from datamodels.models import MachineCapabilities


import numpy as np


from typing import List


class BasicNormalizationTransformation(DataTransformation):
    import numpy as np

    id = "fall-detection-p2-normalization"
    description = """A transformation that normalizes the input numerical data by subtracting the mean
    and dividing by the standard deviation. Since the transformation was developed for the fall detection model in Pilot 2,
    the default values of mean and std are set for that model. However, different parametrization can also be used."""
    parameter_types = {"mean": List[float], "std": List[float]}
    # here the values are in order of x, y, z, svm
    default_values = {"mean": [-1082.6096911969087, -152.6260465746924, -125.43403613436833, 1478.656413757267],
                      "std": [1049.907620444408, 556.0640075934297, 780.0383078816246, 1022.0586009091664]}
    outputs = [np.ndarray]
    needs = MachineCapabilities(preinstalled_libraries={"numpy": "1.23.5"})

    def set_parameters(self, parameters):
        self.params = parameters

    def get_parameters(self):
        return self.params

    def transform_data(self, data):
        data = np.array(data)
        mean = np.array(self.params["mean"])
        std = np.array(self.params["std"])
        return (data - mean)/std

    def transform_format(self, format):
        if "numerical" in format["data_types"]:
            format["data_types"]["numerical"]["normalized"] = True
        return format


class BasicDimensionExpansionTransformation(DataTransformation):
    import numpy as np

    id = "basic-expand-dimensions"
    description = """Basically a wrapper around numpy.expand_dims. 
    Expands the shape of the array by inserting a new axis, that will appear at the axis position in expanded array shape"""
    parameter_types = {"axis": int}
    default_values = {"axis": 0}
    outputs = [np.ndarray]
    needs = MachineCapabilities(preinstalled_libraries={"numpy": "1.23.5"})

    def set_parameters(self, parameters):
        self.params = parameters

    def get_parameters(self):
        return self.params

    def transform_data(self, data):
        data = np.array(data)
        return np.expand_dims(data, axis=self.params["axis"])

    def transform_format(self, format):
        if "numerical" in format["data_types"]:
            axis = self.params["axis"]
            format["data_types"]["numerical"]["size"].insert(axis, 1)
        return format