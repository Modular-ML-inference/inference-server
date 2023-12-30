from data_transformation.transformation import DataTransformation
from datamodels.models import MachineCapabilities


import numpy as np




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
