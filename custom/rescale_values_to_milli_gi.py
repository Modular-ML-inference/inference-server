from data_transformation.transformation import DataTransformation
from datamodels.models import MachineCapabilities


import numpy as np


from typing import List


class RescaleValuesToMilliGi(DataTransformation):
    import numpy as np

    id = "fall-detection-p2-rescale-values-to-milli-gi"
    description = """A transformation that rescale all of the input values according to the multiplier. Here, the multiplier is used to """
    parameter_types = {"multiplier":float}
    # here the values are in order of x, y, z, svm
    default_values = {"multiplier": 8000.0/2048.0}
    outputs = [np.ndarray]
    needs = MachineCapabilities(preinstalled_libraries={"numpy": "1.23.5"})

    def set_parameters(self, parameters):
        self.params = parameters

    def get_parameters(self):
        return self.params

    def transform_data(self, data):
        data = np.array(data)
        return data*self.params["multiplier"]

    def transform_format(self, format):
        if "numerical" in format["data_types"]:
            format["data_types"]["numerical"]["unit"] = "milligraviton"
        return format
